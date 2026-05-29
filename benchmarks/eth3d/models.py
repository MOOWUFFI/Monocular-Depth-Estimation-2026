"""All model class definitions used by the ETH3D evaluator.

This module is self-contained — every architecture and side-channel module
needed to load the project's checkpoints is defined inline so the evaluator
never imports the training packages.

Contents:
  * Baseline models : ConvBlock, BasicBlock, Bottleneck, UNet, ResNetDepth,
                      build_baseline_model
  * Eigen aggregation : color_affinity / semantic_affinity /
                        symmetric_normalized_laplacian / EigenAggregationModule
  * SEGF              : Sobel edge perception-guided filtering module
  * Depth heads       : DepthDecodeHead, ResNet34DepthDecodeHead
  * Encoders          : ResNet34EncoderWrapper
  * EAGLE model       : EAGLEDepthModel
  * TinyDepthUNet     : compact MobileNetV3-Small U-Net student
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# ── Baseline models ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicBlock(nn.Module):
    """ResNet basic block (used for ResNet-18 / 34)."""
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if stride != 1 or in_c != out_c
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    """ResNet bottleneck block (used for ResNet-50)."""
    expansion = 4

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,      out_c,     1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c,     out_c,     3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c * 4,     1, bias=False),
            nn.BatchNorm2d(out_c * 4),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c * 4),
            )
            if stride != 1 or in_c != out_c * 4
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.shortcut(x))


_RESNET_CFG: dict[str, tuple] = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}


def _make_layer(block, in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
    layers = [block(in_c, out_c, stride)]
    for _ in range(1, n):
        layers.append(block(out_c * block.expansion, out_c))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Simple U-Net for monocular depth estimation."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3,   64)
        self.enc2 = ConvBlock(64,  128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 512, dropout=0.3)
        self.up4  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512,  256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256,  128)
        self.up1  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec1 = ConvBlock(128,   64)
        self.out  = nn.Conv2d(64, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return F.softplus(self.out(d1))


class ResNetDepth(nn.Module):
    """ResNet-18/34/50 encoder with a symmetric U-Net-style decoder."""

    def __init__(self, variant: str):
        super().__init__()
        block, layers = _RESNET_CFG[variant]
        exp = block.expansion

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(block,  64,       64, layers[0], stride=1)
        self.layer2 = _make_layer(block,  64*exp,  128, layers[1], stride=2)
        self.layer3 = _make_layer(block, 128*exp,  256, layers[2], stride=2)
        self.layer4 = _make_layer(block, 256*exp,  512, layers[3], stride=2)

        c = [64*exp, 128*exp, 256*exp, 512*exp]
        self.bottleneck = ConvBlock(c[3], c[3], dropout=0.3)

        self.up4  = nn.ConvTranspose2d(c[3],    c[3]//2, 2, stride=2)
        self.dec4 = ConvBlock(c[3]//2 + c[2],   c[3]//2)
        self.up3  = nn.ConvTranspose2d(c[3]//2, c[2]//2, 2, stride=2)
        self.dec3 = ConvBlock(c[2]//2 + c[1],   c[2]//2)
        self.up2  = nn.ConvTranspose2d(c[2]//2, c[1]//2, 2, stride=2)
        self.dec2 = ConvBlock(c[1]//2 + c[0],   c[1]//2)
        self.up1a = nn.ConvTranspose2d(c[1]//2, c[0]//2, 2, stride=2)
        self.dec1 = ConvBlock(c[0]//2,           c[0]//2)
        self.up1b = nn.ConvTranspose2d(c[0]//2, c[0]//4, 2, stride=2)
        self.dec0 = ConvBlock(c[0]//4,           c[0]//4)
        self.out  = nn.Conv2d(c[0]//4, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stem(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(torch.cat([self.up4(b),   s3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4),  s2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3),  s1], 1))
        d1 = self.dec1(self.up1a(d2))
        d0 = self.dec0(self.up1b(d1))
        return F.softplus(self.out(d0))


_VARIANTS = {"unet", "resnet18", "resnet34", "resnet50"}


def build_baseline_model(variant: str) -> nn.Module:
    """Return an uninitialized model for the given variant string."""
    variant = variant.lower()
    if variant == "unet":
        return UNet()
    if variant in _RESNET_CFG:
        return ResNetDepth(variant)
    raise ValueError(
        f"Unknown baseline variant '{variant}'. "
        f"Expected one of: {sorted(_VARIANTS)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── Eigen Aggregation Module (EAGLE side-channel) ─────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reference: Kim et al., "EAGLE: Eigen Aggregation Learning for Object-Centric
# Unsupervised Semantic Segmentation", CVPR 2024. The module computes a
# patch-level affinity graph, its symmetric normalized Laplacian, the k
# smallest non-trivial eigenvectors, and a soft cluster assignment. It is a
# side-channel that (by default) does not modify the feature map fed to the
# decoder.

def color_affinity(rgb_low: torch.Tensor, sigma_color: float = 0.2) -> torch.Tensor:
    """Gaussian (RBF) color affinity over every patch pair.

    rgb_low: [B, 3, h, w] in [0, 1] (raw RGB, not ImageNet-normalized).
    Returns: [B, N, N] with N = h * w.
    """
    B, C, h, w = rgb_low.shape
    N = h * w
    c = rgb_low.reshape(B, C, N).permute(0, 2, 1)
    diff = c.unsqueeze(2) - c.unsqueeze(1)
    sq = (diff * diff).sum(-1)
    return torch.exp(-sq / (2.0 * sigma_color * sigma_color + 1e-8))


def semantic_affinity(feat: torch.Tensor) -> torch.Tensor:
    """A_seg = S S^T on L2-normalised features."""
    B, C, h, w = feat.shape
    N = h * w
    f = feat.reshape(B, C, N).permute(0, 2, 1)
    f = F.normalize(f, dim=-1, eps=1e-6)
    return f @ f.transpose(1, 2)


def symmetric_normalized_laplacian(W: torch.Tensor) -> torch.Tensor:
    """L_sym = I - D^{-1/2} W D^{-1/2} for a non-negative affinity W."""
    d = W.sum(dim=-1).clamp_min(1e-6)
    d_inv_sqrt = d.rsqrt()
    W_norm = W * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1)
    eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype).expand_as(W)
    L = eye - W_norm
    # Defensively symmetrize against fp16 / numerical noise.
    return 0.5 * (L + L.transpose(-1, -2))


class EigenAggregationModule(nn.Module):
    """Compute EiCue (eigenvector-derived cluster assignment) at one spatial scale.

    Returns the (optionally residual-aggregated) feature, the eigenvectors U,
    the eigenvalues, and the soft cluster-assignment logits. The decoder
    consumes the unmodified `feat` unless `use_residual` is set.
    """

    def __init__(
        self,
        channels: int,
        k: int = 4,
        num_clusters: int = 10,
        sigma_color: float = 0.2,
        drop_first: int = 1,
        use_residual: bool = False,
        differentiable_eigh: bool = False,
        eigh_eps_reg: float = 1e-2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.k = k
        self.num_clusters = num_clusters
        self.sigma_color = sigma_color
        self.drop_first = drop_first
        self.use_residual = use_residual
        self.differentiable_eigh = differentiable_eigh
        self.eigh_eps_reg = float(eigh_eps_reg)

        # proj_in transforms `feat` into the space used for semantic affinity.
        self.proj_in = nn.Conv2d(channels, channels, 1, bias=False)

        # Cluster centers C ∈ R^{k × num_clusters}, learned end-to-end.
        self.cluster_centers = nn.Parameter(
            torch.randn(k, num_clusters) * (1.0 / math.sqrt(k))
        )

        # Optional architectural residual aggregation:
        #     out_feat = feat + proj_out(U U^T feat)
        # Zero-init proj_out so the module starts as a passthrough.
        if use_residual:
            self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.zeros_(self.proj_out.weight)
        else:
            self.proj_out = None

    def _eigh_safe(self, L: torch.Tensor):
        """eigh wrapper. Always fp32; in differentiable mode adds ε·I
        spectral regularization, jitter-retries on forward failure, and
        registers a NaN-zeroing backward hook on the eigvecs/eigvals so
        a numerically dodgy batch doesn't poison the optimizer step."""
        orig_dtype = L.dtype
        with torch.amp.autocast(device_type="cuda", enabled=False):
            L32 = L.float()
            if self.differentiable_eigh and self.eigh_eps_reg > 0:
                eye = torch.eye(
                    L32.shape[-1], device=L32.device, dtype=L32.dtype
                ).expand_as(L32)
                L32 = L32 + self.eigh_eps_reg * eye
            try:
                eigvals, eigvecs = torch.linalg.eigh(L32)
            except Exception:
                # Failure typically means ill-conditioned L (close eigvals).
                # Add an order-of-magnitude more jitter and try again.
                jitter = max(self.eigh_eps_reg * 10.0, 1e-2)
                eye = torch.eye(
                    L32.shape[-1], device=L32.device, dtype=L32.dtype
                ).expand_as(L32)
                eigvals, eigvecs = torch.linalg.eigh(L32 + jitter * eye)
        if self.differentiable_eigh:
            def _nan_zero(grad):
                if grad is None:
                    return None
                return torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
            eigvecs.register_hook(_nan_zero)
            eigvals.register_hook(_nan_zero)
        else:
            eigvals = eigvals.to(orig_dtype)
            eigvecs = eigvecs.to(orig_dtype)
        return eigvals, eigvecs

    # Back-compat alias.
    _eigh_fp32 = _eigh_safe

    def forward(
        self,
        feat: torch.Tensor,
        rgb_low: torch.Tensor,
        affinity_feat: torch.Tensor | None = None,
    ):
        """feat: [B, C, h, w]. rgb_low: [B, 3, h, w] in [0, 1].

        `affinity_feat` (optional): [B, C, h, w] — features used to build
        the SEMANTIC affinity. Defaults to `feat` if not provided.
        """
        B, C, h, w = feat.shape
        N = h * w

        affinity_in = affinity_feat if affinity_feat is not None else feat
        f_proj = self.proj_in(affinity_in)
        K_sem = semantic_affinity(f_proj).clamp_min(0.0)
        K_col = color_affinity(rgb_low, self.sigma_color).to(K_sem.dtype)
        W = K_sem + K_col

        L = symmetric_normalized_laplacian(W)
        eigvals_full, eigvecs_full = self._eigh_safe(L)

        start = self.drop_first
        end = start + self.k
        if self.differentiable_eigh:
            U = eigvecs_full[:, :, start:end].to(feat.dtype)
            eigvals = eigvals_full[:, start:end].to(feat.dtype)
        else:
            U = eigvecs_full[:, :, start:end].detach()
            eigvals = eigvals_full[:, start:end].detach()

        # Soft cluster assignment in eigenvector space.
        logits = U @ self.cluster_centers.to(U.dtype)

        if self.use_residual:
            f_flat = feat.reshape(B, C, N)
            coeffs = torch.einsum("bnk,bcn->bck", U, f_flat)
            f_eig_flat = torch.einsum("bnk,bck->bcn", U, coeffs)
            f_eig = f_eig_flat.reshape(B, C, h, w)
            out_feat = feat + self.proj_out(f_eig)
        else:
            out_feat = feat

        return out_feat, U, eigvals, logits


# ═══════════════════════════════════════════════════════════════════════════════
# ── SEGF: Sobel Edge Perception-Guided Filtering ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# Adapted from Tian et al., "MsGf: A Lightweight Self-Supervised Monocular
# Depth Estimation Framework with Multi-Scale Feature Extraction"
# (Sensors 2025, 25(20), 6380). Spatially-adaptive filtering of a feature
# map whose filter kernel is generated from Sobel-detected RGB edges.

class SEGF(nn.Module):
    """Sobel Edge Perception-Guided Filtering."""

    # Direction-enhanced Sobel: heavier central element than vanilla 1-2-1.
    _SOBEL_X = torch.tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]) / 16.0
    _SOBEL_Y = torch.tensor([[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]]) / 16.0

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.channels = channels
        self.k = kernel_size

        self.register_buffer("sobel_x", self._SOBEL_X.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", self._SOBEL_Y.view(1, 1, 3, 3))

        # Project RGB → C channels at feat resolution.
        self.guide_proj = nn.Conv2d(3, channels, 3, padding=1, bias=False)

        # Fuse horizontal + vertical Sobel responses (concat along channel).
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            nn.PReLU(channels),
        )
        self.proj_hor = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.proj_ver = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # Kernel generators: produce k² values per pixel for each direction.
        self.gen_h = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, self.k * self.k, 1, bias=False),
        )
        self.gen_v = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, self.k * self.k, 1, bias=False),
        )

        # Output projection — ZERO-INIT so SEGF starts as the IDENTITY.
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Learnable scale factor for kernel normalisation.
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """feat: [B, C, h, w]. rgb: [B, 3, H, W] in [0, 1].

        Returns: feat + per-pixel-filtered residual, [B, C, h, w].
        """
        B, C, h, w = feat.shape
        orig_dtype = feat.dtype

        with torch.amp.autocast(device_type="cuda", enabled=False):
            feat32 = feat.float()
            rgb32 = rgb.float()
            if rgb32.shape[-2:] != (h, w):
                rgb32 = F.interpolate(rgb32, size=(h, w),
                                      mode="bilinear", align_corners=False)
            g = self.guide_proj(rgb32)                          # [B, C, h, w]

            sobel_x = self.sobel_x.expand(C, 1, 3, 3)
            sobel_y = self.sobel_y.expand(C, 1, 3, 3)
            fx = F.conv2d(g, sobel_x, padding=1, groups=C)
            fy = F.conv2d(g, sobel_y, padding=1, groups=C)

            f_fused = self.fuse(torch.cat([fx, fy], dim=1))
            f_hor = self.proj_hor(f_fused)
            f_ver = self.proj_ver(f_fused)

            k_h = torch.tanh(self.gen_h(f_hor))                 # [B, k², h, w]
            k_v = torch.tanh(self.gen_v(f_ver))

            k_fusion = (k_h * k_v) / (self.alpha.abs() + 1e-2)
            k_soft = F.softmax(k_fusion, dim=1)                 # [B, k², h, w]

            feat_unf = F.unfold(feat32, kernel_size=self.k,
                                padding=self.k // 2)            # [B, C·k², h*w]
            feat_unf = feat_unf.view(B, C, self.k * self.k, h, w)
            filtered = (feat_unf * k_soft.unsqueeze(1)).sum(dim=2)
            filtered = self.out_proj(filtered)

        return feat + filtered.to(orig_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# ── ResNet34 encoder + ASPP depth head ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

_RESNET34_HIDDEN_SIZES: List[int] = [64, 128, 256, 512]


class ResNet34EncoderWrapper(nn.Module):
    """Wraps torchvision ResNet-34 to emit a ``hidden_states`` tuple that
    matches the HuggingFace encoder output contract used by EAGLEDepthModel.

    Stages (with 256×256 input):
      Stage 0: C=64,  stride=4
      Stage 1: C=128, stride=8
      Stage 2: C=256, stride=16
      Stage 3: C=512, stride=32
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as _tv_models
        weights = _tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = _tv_models.resnet34(weights=weights)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True, **kwargs):
        x  = self.stem(pixel_values)
        h0 = self.layer1(x)
        h1 = self.layer2(h0)
        h2 = self.layer3(h1)
        h3 = self.layer4(h2)

        return SimpleNamespace(hidden_states=(h0, h1, h2, h3))

    def save_pretrained(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str, pretrained: bool = True) -> "ResNet34EncoderWrapper":
        import os
        model = cls(pretrained=pretrained)
        w = f"{path}/pytorch_model.bin"
        if os.path.exists(w):
            model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        return model


class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling used by ResNet34DepthDecodeHead."""

    def __init__(self, in_ch: int, out_ch: int, atrous_rates: tuple = (2, 4, 6)) -> None:
        super().__init__()
        branches: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            branches.append(_ASPPConv(in_ch, out_ch, rate))
        branches.append(_ASPPPooling(in_ch, out_ch))
        self.convs   = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class ResNet34DepthDecodeHead(nn.Module):
    """DepthDecodeHead variant with optional ASPP on the deepest encoder stage.

    Used by checkpoints produced with a ``use_aspp`` flag. When
    ``use_aspp=False`` the architecture is identical to ``DepthDecodeHead``.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int = 256,
        dropout: float = 0.1,
        use_aspp: bool = False,
    ) -> None:
        super().__init__()
        self.num_stages = len(hidden_sizes)
        self.use_aspp   = use_aspp

        self.linear_c = nn.ModuleList()
        for i, c in enumerate(hidden_sizes):
            if use_aspp and i == self.num_stages - 1:
                self.linear_c.append(nn.Identity())
            else:
                self.linear_c.append(nn.Linear(c, decoder_hidden_size))

        if use_aspp:
            self.aspp = _ASPP(in_ch=hidden_sizes[-1], out_ch=decoder_hidden_size)

        self.linear_fuse = nn.Conv2d(
            self.num_stages * decoder_hidden_size, decoder_hidden_size, 1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout(p=dropout)
        self.depth_head = nn.Conv2d(decoder_hidden_size, 1, 1)

    def forward(
        self,
        encoder_hidden_states: Tuple[torch.Tensor, ...],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        ref_size = encoder_hidden_states[0].shape[-2:]
        upsampled: List[torch.Tensor] = []

        for i, (x, linear) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            if self.use_aspp and i == self.num_stages - 1:
                x_proj = self.aspp(x)
            else:
                B, C, H, W = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
                x_proj = linear(x_flat).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if x_proj.shape[-2:] != ref_size:
                x_proj = F.interpolate(x_proj, size=ref_size, mode="bilinear", align_corners=False)
            upsampled.append(x_proj)

        x = torch.cat(upsampled, dim=1)
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.depth_head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.softplus(x).squeeze(1)  # (B, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Depth decode head ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class DepthDecodeHead(nn.Module):
    """All-MLP depth decode head (SegFormer-style, single-channel metric depth).

    For each encoder stage i:
      (B, C_i, H_i, W_i) → Linear(C_i → D) → bilinear upsample to stage-0 res

    Concatenate all stages in sequential order (stage 0 first) →
    fuse Conv(4D→D) + BN + ReLU → depth Conv(D→1)
    → bilinear upsample to input res → Softplus (positive depth).

    NOTE: uses sequential concat so depth-head weights from the
    SFT checkpoints load correctly. Do NOT use checkpoints that use a
    reversed ([::-1]) concat.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_stages = len(hidden_sizes)
        self.linear_c = nn.ModuleList(
            [nn.Linear(c, decoder_hidden_size) for c in hidden_sizes]
        )
        self.linear_fuse = nn.Conv2d(
            self.num_stages * decoder_hidden_size,
            decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout(p=dropout)
        self.depth_head = nn.Conv2d(decoder_hidden_size, 1, kernel_size=1)

    def forward(
        self,
        encoder_hidden_states: Tuple[torch.Tensor, ...],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        ref_size = encoder_hidden_states[0].shape[-2:]
        upsampled: List[torch.Tensor] = []
        for x, linear in zip(encoder_hidden_states, self.linear_c):
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_proj = linear(x_flat)
            x_proj = (
                x_proj.reshape(B, H, W, -1)
                       .permute(0, 3, 1, 2)
                       .contiguous()
            )
            if (H, W) != ref_size:
                x_proj = F.interpolate(
                    x_proj, size=ref_size, mode="bilinear", align_corners=False
                )
            upsampled.append(x_proj)

        x = torch.cat(upsampled, dim=1)  # sequential order: stage 0 first
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.depth_head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.softplus(x).squeeze(1)  # (B, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# ── EAGLE depth model ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class EAGLEDepthModel(nn.Module):
    """SegFormer / MobileViT encoder + DepthDecodeHead + EigenAggregationModules.

    The EAMs are *side-channel* modules — they do NOT modify the feature maps
    fed to the depth decoder. For each requested encoder stage s the EAM:

      1. Builds a patch-level affinity graph: A = A_color + A_seg
      2. Computes the symmetric normalized graph Laplacian L_sym.
      3. Takes the k smallest non-trivial eigenvectors V̂ of L_sym.
      4. Computes soft cluster assignment logits P = V̂ · C.
    """

    def __init__(
        self,
        encoder,
        depth_head: DepthDecodeHead,
        hidden_sizes: List[int],
        eam_stages: List[int],
        eam_k: int = 4,
        num_clusters: int = 10,
        eam_sigma_color: float = 0.2,
        multi_layer_affinity: bool = False,
        differentiable_eigh: bool = False,
        eigh_eps_reg: float = 1e-2,
        input_is_imagenet_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.encoder      = encoder
        self.depth_head   = depth_head
        self.eam_stages   = sorted(eam_stages)
        self.hidden_sizes = hidden_sizes
        self.multi_layer_affinity = multi_layer_affinity
        self.input_is_imagenet_normalized = input_is_imagenet_normalized

        self.eams = nn.ModuleDict({
            str(s): EigenAggregationModule(
                channels=hidden_sizes[s],
                k=eam_k,
                num_clusters=num_clusters,
                sigma_color=eam_sigma_color,
                use_residual=False,
                differentiable_eigh=differentiable_eigh,
                eigh_eps_reg=eigh_eps_reg,
            )
            for s in self.eam_stages
        })

        if multi_layer_affinity:
            self.affinity_projs = nn.ModuleDict()
            for s in self.eam_stages:
                src_stages = [max(0, s - 2), max(0, s - 1), s]
                seen: list[int] = []
                for si in src_stages:
                    if si not in seen:
                        seen.append(si)
                concat_ch = sum(hidden_sizes[si] for si in seen)
                if concat_ch != hidden_sizes[s]:
                    self.affinity_projs[str(s)] = nn.Conv2d(
                        concat_ch, hidden_sizes[s], 1, bias=False
                    )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _build_affinity_feat(
        self,
        s: int,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Optional[torch.Tensor]:
        if not self.multi_layer_affinity:
            return None

        src_stages: list[int] = []
        for si in [max(0, s - 2), max(0, s - 1), s]:
            if si not in src_stages:
                src_stages.append(si)

        target_size = hidden_states[s].shape[-2:]
        parts = []
        for si in src_stages:
            hi = hidden_states[si]
            if hi.shape[-2:] != target_size:
                hi = F.interpolate(hi, size=target_size, mode="bilinear", align_corners=False)
            parts.append(hi)

        concat = torch.cat(parts, dim=1)
        proj_key = str(s)
        if proj_key in self.affinity_projs:
            return self.affinity_projs[proj_key](concat)
        return concat

    def forward(self, pixel_values: torch.Tensor) -> Dict:
        """
        Args:
            pixel_values: (B, 3, H, W)

        Returns a dict with:
            depth   : (B, H, W)
            logits  : dict[int, (B, N, K)]
            Us      : dict[int, (B, N, k)]
            eigvals : dict[int, (B, k)]
        """
        enc_out       = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = enc_out.hidden_states
        H, W          = pixel_values.shape[-2:]

        if self.input_is_imagenet_normalized:
            rgb_raw = (
                pixel_values * self.imagenet_std + self.imagenet_mean
            ).clamp(0.0, 1.0)
        else:
            rgb_raw = pixel_values.clamp(0.0, 1.0)

        logits_dict: Dict[int, torch.Tensor]  = {}
        Us_dict:     Dict[int, torch.Tensor]  = {}
        eigvals_dict: Dict[int, torch.Tensor] = {}

        for s in self.eam_stages:
            feat   = hidden_states[s]
            rgb_lo = F.interpolate(
                rgb_raw, size=feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            affinity_feat = self._build_affinity_feat(s, hidden_states)

            _, U, lam, lg = self.eams[str(s)](feat, rgb_lo, affinity_feat=affinity_feat)
            logits_dict[s]  = lg
            Us_dict[s]      = U
            eigvals_dict[s] = lam

        depth = self.depth_head(hidden_states, target_size=(H, W))

        return {
            "depth":   depth,
            "logits":  logits_dict,
            "Us":      Us_dict,
            "eigvals": eigvals_dict,
        }

    def param_groups(
        self,
        base_lr: float,
        encoder_lr_mult: float,
        weight_decay: float,
    ) -> List[dict]:
        """Return AdamW param groups with differential LR for encoder vs decoder+EAM."""
        enc_wd, enc_nd, oth_wd, oth_nd = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_enc = name.startswith("encoder.")
            no_wd  = p.ndim == 1 or name.endswith(".bias")
            if is_enc and no_wd:
                enc_nd.append(p)
            elif is_enc:
                enc_wd.append(p)
            elif no_wd:
                oth_nd.append(p)
            else:
                oth_wd.append(p)
        return [
            {"params": enc_wd,  "lr": base_lr * encoder_lr_mult, "weight_decay": weight_decay},
            {"params": enc_nd,  "lr": base_lr * encoder_lr_mult, "weight_decay": 0.0},
            {"params": oth_wd,  "lr": base_lr,                   "weight_decay": weight_decay},
            {"params": oth_nd,  "lr": base_lr,                   "weight_decay": 0.0},
        ]


# Backward-compatible alias for scripts that import SegFormerEAGLEModel.
SegFormerEAGLEModel = EAGLEDepthModel


# ═══════════════════════════════════════════════════════════════════════════════
# ── TinyDepthUNet (MobileNetV3-Small student) ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# Compact U-Net depth student with an ImageNet-pretrained MobileNetV3-Small
# encoder (stem patched to 5-ch RGB+XY), an ASPP-lite bottleneck (with an
# optional side-channel EAGLE EAM hook), a U-Net decoder, and a 1x1 head
# producing direct log-depth. Inlined here so the evaluator stays
# self-contained.

_TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ASPPLite(nn.Module):
    """Tiny ASPP at the bottleneck: 1x1 + two dilated 3x3 + image pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.gap_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        gp = F.adaptive_avg_pool2d(x, 1)
        gp = self.gap_conv(gp)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        cat = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), gp], dim=1)
        return self.project(cat)


class _UpBlock(nn.Module):
    """ConvTranspose 2x + concat skip + 3x3 conv + GN + ReLU + 3x3 conv + GN + ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False)
        in_after = in_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_after, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TinyDepthUNet(nn.Module):
    """Compact U-Net depth student. ~1.5-2 M params; emits direct log-depth."""

    def __init__(
        self,
        bottleneck_channels: int = 64,
        decoder_channels: tuple[int, int, int, int] = (64, 48, 32, 16),
        eam_scales: tuple[int, ...] = (),
        eam_k: int = 4,
        num_clusters: int = 10,
        eam_sigma_color: float = 0.2,
        use_residual_eam: bool = False,
        pretrained_encoder: bool = True,
        differentiable_eigh: bool = False,
        multi_layer_eam: bool = False,
        use_segf: bool = False,
        segf_kernel_size: int = 3,
        use_aspp: bool = True,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        mbn = models.mobilenet_v3_small(weights=weights)
        # mbn.features structure (verified for 512x512 input):
        #   [0]   stem  stride 2,  16 ch
        #   [1]   bn1   stride 4,  16 ch       <-- stride-4 skip
        #   [2-3] bn2   stride 8,  24 ch       <-- stride-8 skip at [3]
        #   [4-8] bn3   stride 16, 40->48 ch   <-- stride-16 skip at [8]
        #   [9-11] bn4  stride 32, 96 ch       <-- bottleneck input at [11]
        #   [12]  expand stride 32, 576 ch     (not used)

        # Patch stem conv from 3-ch to 5-ch (RGB + X/Y coord grids).
        old_stem_conv = mbn.features[0][0]
        new_stem_conv = nn.Conv2d(
            5, old_stem_conv.out_channels,
            kernel_size=old_stem_conv.kernel_size,
            stride=old_stem_conv.stride,
            padding=old_stem_conv.padding,
            bias=(old_stem_conv.bias is not None),
        )
        with torch.no_grad():
            new_stem_conv.weight[:, :3, :, :] = old_stem_conv.weight
            new_stem_conv.weight[:, 3:, :, :] = 0.0
            if old_stem_conv.bias is not None:
                new_stem_conv.bias[:] = old_stem_conv.bias
        mbn.features[0][0] = new_stem_conv

        # Split the encoder into stage modules so we can grab skips cleanly.
        self.enc_stem = mbn.features[0]      # -> stride 2
        self.enc_s4 = mbn.features[1]        # -> stride 4 (16 ch)  : skip
        self.enc_s8 = mbn.features[2:4]      # -> stride 8 (24 ch)  : skip after stage
        self.enc_s16 = mbn.features[4:9]     # -> stride 16 (48 ch) : skip after stage
        self.enc_s32 = mbn.features[9:12]    # -> stride 32 (96 ch) : bottleneck input
        self.enc_s8 = nn.Sequential(*list(self.enc_s8))
        self.enc_s16 = nn.Sequential(*list(self.enc_s16))
        self.enc_s32 = nn.Sequential(*list(self.enc_s32))

        # ASPP-lite at the bottleneck: 96 -> bottleneck_channels.
        # When `use_aspp=False`, replace with a plain 1×1 conv to keep the
        # rest of the architecture identical (used for ablations).
        self.use_aspp = bool(use_aspp)
        if self.use_aspp:
            self.aspp = _ASPPLite(96, bottleneck_channels)
        else:
            self.aspp = nn.Sequential(
                nn.Conv2d(96, bottleneck_channels, 1, bias=False),
                nn.GroupNorm(min(8, bottleneck_channels), bottleneck_channels),
                nn.ReLU(inplace=True),
            )

        # EAM hook at the bottleneck stride (32). Off by default.
        self.eam_scales = tuple(sorted(set(eam_scales), reverse=True))
        for s in self.eam_scales:
            if s != 32:
                raise ValueError(
                    f"TinyDepthUNet currently only supports EAM at stride 32; got {s}."
                )
        self.eams = nn.ModuleDict({
            str(s): EigenAggregationModule(
                channels=bottleneck_channels,
                k=eam_k,
                num_clusters=num_clusters,
                sigma_color=eam_sigma_color,
                use_residual=use_residual_eam,
                differentiable_eigh=differentiable_eigh,
            )
            for s in self.eam_scales
        })
        self.differentiable_eigh = differentiable_eigh

        # Multi-layer affinity: project skip features from strides 8 + 16 to
        # `bottleneck_channels`, downsample to the stride-32 EAM grid, concat
        # with the stride-32 bottleneck feature, and pipe through a final 1×1
        # to recover the original bottleneck dim.
        self.multi_layer_eam = bool(multi_layer_eam and (32 in self.eam_scales))
        if self.multi_layer_eam:
            self.eam_skip_proj_s8 = nn.Conv2d(24, bottleneck_channels, 1, bias=False)
            self.eam_skip_proj_s16 = nn.Conv2d(48, bottleneck_channels, 1, bias=False)
            self.eam_multi_proj = nn.Conv2d(3 * bottleneck_channels, bottleneck_channels, 1, bias=False)

        # Optional SEGF block at the bottleneck, applied after the EAM hook
        # (if any) and before the decoder.
        self.use_segf = bool(use_segf)
        if self.use_segf:
            self.segf = SEGF(channels=bottleneck_channels, kernel_size=segf_kernel_size)

        # Decoder: 4 up-blocks (stride 32->16->8->4->2 + final upsample).
        c1, c2, c3, c4 = decoder_channels
        self.up1 = _UpBlock(bottleneck_channels, skip_ch=48, out_ch=c1)  # 32 -> 16
        self.up2 = _UpBlock(c1, skip_ch=24, out_ch=c2)                   # 16 -> 8
        self.up3 = _UpBlock(c2, skip_ch=16, out_ch=c3)                   # 8 -> 4
        self.up4 = _UpBlock(c3, skip_ch=0, out_ch=c4)                    # 4 -> 2

        # Head: 1x1 -> log-depth.
        self.head = nn.Conv2d(c4, 1, 1)
        nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="linear")
        with torch.no_grad():
            self.head.weight.mul_(0.01)
        nn.init.zeros_(self.head.bias)

        # Normalization buffers (for rgb_low denormalization).
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(_TINY_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(_TINY_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _coord_channels(self, rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb.shape
        y = torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, y], dim=1)

    def forward(self, rgb: torch.Tensor) -> dict:
        b, _, h, w = rgb.shape
        coords = self._coord_channels(rgb)
        x_in = torch.cat([rgb, coords], dim=1)

        # --- encoder ----------------------------------------------------
        x2 = self.enc_stem(x_in)        # stride 2,  16 ch
        s4 = self.enc_s4(x2)            # stride 4,  16 ch
        s8 = self.enc_s8(s4)            # stride 8,  24 ch
        s16 = self.enc_s16(s8)          # stride 16, 48 ch
        s32 = self.enc_s32(s16)         # stride 32, 96 ch

        # --- bottleneck + optional EAM ---------------------------------
        bottleneck = self.aspp(s32)     # stride 32, bottleneck_channels

        Us: dict[int, torch.Tensor] = {}
        eigvals: dict[int, torch.Tensor] = {}
        logits: dict[int, torch.Tensor] = {}
        rgb_low = (rgb * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        if 32 in self.eam_scales:
            rgb_at_eam = F.interpolate(
                rgb_low, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False
            )
            affinity_feat = None
            if self.multi_layer_eam:
                target_size = bottleneck.shape[-2:]
                s8_proj = self.eam_skip_proj_s8(s8)
                s16_proj = self.eam_skip_proj_s16(s16)
                s8_d = F.adaptive_avg_pool2d(s8_proj, target_size)
                s16_d = F.adaptive_avg_pool2d(s16_proj, target_size)
                concat = torch.cat([s8_d, s16_d, bottleneck], dim=1)
                affinity_feat = self.eam_multi_proj(concat)
            bottleneck, U, lam, lg = self.eams["32"](
                bottleneck, rgb_at_eam, affinity_feat=affinity_feat
            )
            Us[32] = U; eigvals[32] = lam; logits[32] = lg

        if self.use_segf:
            bottleneck = self.segf(bottleneck, rgb_low)

        bottleneck_feat = bottleneck  # for KD feature loss

        # --- decoder ----------------------------------------------------
        d1 = self.up1(bottleneck, s16)  # -> stride 16
        d2 = self.up2(d1, s8)           # -> stride 8
        d3 = self.up3(d2, s4)           # -> stride 4
        d4 = self.up4(d3, None)         # -> stride 2
        log_depth = self.head(d4)       # stride 2, 1ch
        log_depth = F.interpolate(log_depth, size=(h, w), mode="bilinear", align_corners=False)
        disp = torch.exp(-log_depth)    # for legacy disparity-space losses/viz

        rgb_low_eam = F.interpolate(
            rgb_low, size=bottleneck_feat.shape[-2:], mode="bilinear", align_corners=False
        )

        return {
            "log_depth": log_depth,
            "disp": disp,
            "bottleneck_feat": bottleneck_feat,
            "Us": Us,
            "eigvals": eigvals,
            "logits": logits,
            "rgb_low": rgb_low_eam,
        }

    def param_groups(self, base_lr: float, encoder_mult: float, weight_decay: float):
        """Encoder stages at `encoder_mult * lr`, rest at `lr`. No WD on norms / biases."""
        encoder_modules = (self.enc_stem, self.enc_s4, self.enc_s8, self.enc_s16, self.enc_s32)
        encoder_param_ids = {id(p) for m in encoder_modules for p in m.parameters()}
        enc_decay, enc_nodecay, oth_decay, oth_nodecay = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_encoder = id(p) in encoder_param_ids
            no_decay = p.ndim == 1 or name.endswith(".bias")
            if is_encoder and no_decay:
                enc_nodecay.append(p)
            elif is_encoder:
                enc_decay.append(p)
            elif no_decay:
                oth_nodecay.append(p)
            else:
                oth_decay.append(p)
        return [
            {"params": enc_decay,    "lr": base_lr * encoder_mult, "weight_decay": weight_decay},
            {"params": enc_nodecay,  "lr": base_lr * encoder_mult, "weight_decay": 0.0},
            {"params": oth_decay,    "lr": base_lr,                "weight_decay": weight_decay},
            {"params": oth_nodecay,  "lr": base_lr,                "weight_decay": 0.0},
        ]
