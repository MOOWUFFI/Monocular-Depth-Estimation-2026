"""Depth model: ResNet34 encoder + DepthDecodeHead (optional ASPP) +
per-stage EigenAggregationModules (EAGLE).

`EAGLEDepthModel.forward` returns a dict with the predicted depth plus the EAM
cluster logits / eigenbasis used by the EAGLE losses. Depth is always produced
by the decode head from the raw encoder features; the EAMs only feed the
clustering losses, so an empty `eam_stages` disables EAGLE entirely.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from approaches.Resnet.eagle import EigenAggregationModule

# ResNet34 hidden sizes per encoder stage (layers 1 to 4).
_RESNET34_HIDDEN_SIZES = [64, 128, 256, 512]


# ── ResNet34 wrapper ──────────────────────────────────────────────────────────

class ResNet34EncoderWrapper(nn.Module):
    """Wraps torchvision ResNet34 to emit a HuggingFace-style `hidden_states`
    tuple (layers 1 through 4)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1  # Stage 0: C=64,  stride=4
        self.layer2 = base_model.layer2  # Stage 1: C=128, stride=8
        self.layer3 = base_model.layer3  # Stage 2: C=256, stride=16
        self.layer4 = base_model.layer4  # Stage 3: C=512, stride=32

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True):
        x = self.stem(pixel_values)
        h0 = self.layer1(x)
        h1 = self.layer2(h0)
        h2 = self.layer3(h1)
        h3 = self.layer4(h2)

        class EncoderOutput:
            def __init__(self, states):
                self.hidden_states = states

        return EncoderOutput((h0, h1, h2, h3))

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path_or_name, pretrained: bool = True):
        model = cls(pretrained=pretrained)
        if os.path.isdir(path_or_name):
            weight_path = os.path.join(path_or_name, "pytorch_model.bin")
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        return model


# ── ASPP ──────────────────────────────────────────────────────────────────────

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(2, 4, 6)):
        super().__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


# ── DepthDecodeHead ─────────────────────────────────────────────────────────

class DepthDecodeHead(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int = 256,
        dropout: float = 0.1,
        use_aspp: bool = False,
    ) -> None:
        super().__init__()
        self.num_stages = len(hidden_sizes)
        self.use_aspp = use_aspp

        self.linear_c = nn.ModuleList()
        for i, c in enumerate(hidden_sizes):
            if self.use_aspp and i == self.num_stages - 1:
                self.linear_c.append(nn.Identity())
            else:
                self.linear_c.append(nn.Linear(c, decoder_hidden_size))

        if self.use_aspp:
            self.aspp = ASPP(in_channels=hidden_sizes[-1], out_channels=decoder_hidden_size)

        self.linear_fuse = nn.Conv2d(
            self.num_stages * decoder_hidden_size, decoder_hidden_size, kernel_size=1, bias=False,
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

        for i, (x, linear) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            if self.use_aspp and i == self.num_stages - 1:
                x_proj = self.aspp(x)
            else:
                B, C, H, W = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
                x_proj = linear(x_flat)
                x_proj = x_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if x_proj.shape[-2:] != ref_size:
                x_proj = F.interpolate(x_proj, size=ref_size, mode="bilinear", align_corners=False)
            upsampled.append(x_proj)

        x = torch.cat(upsampled, dim=1)  # sequential order: stage 0 first
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.depth_head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.softplus(x).squeeze(1)  # (B, H, W)


# ── EAGLE-augmented depth model ─────────────────────────────────────────────

class EAGLEDepthModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
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
        enc_out       = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = enc_out.hidden_states   # 4 × (B, C_i, H_i, W_i)
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


# ── Build helper ──────────────────────────────────────────────────────────────

def build_model(
    eam_stages: Optional[List[int]] = None,
    pretrained_encoder: bool = True,
    use_aspp: bool = False,
    decoder_hidden_size: int = 256,
    decoder_dropout: float = 0.1,
    eam_k: int = 4,
    num_clusters: int = 10,
    eam_sigma_color: float = 0.2,
    multi_layer_affinity: bool = False,
    differentiable_eigh: bool = False,
    encoder: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> EAGLEDepthModel:
    """Construct a ResNet34 + DepthDecodeHead (+ EAGLE) model.

    Pass a prebuilt `encoder` to reuse loaded weights; otherwise a fresh
    ResNet34 encoder is created (ImageNet-pretrained if `pretrained_encoder`).
    """
    hidden_sizes = _RESNET34_HIDDEN_SIZES
    stages = sorted(eam_stages) if eam_stages else []

    if encoder is None:
        encoder = ResNet34EncoderWrapper(pretrained=pretrained_encoder)

    depth_head = DepthDecodeHead(
        hidden_sizes=hidden_sizes,
        decoder_hidden_size=decoder_hidden_size,
        dropout=decoder_dropout,
        use_aspp=use_aspp,
    )

    model = EAGLEDepthModel(
        encoder=encoder,
        depth_head=depth_head,
        hidden_sizes=hidden_sizes,
        eam_stages=stages,
        eam_k=eam_k,
        num_clusters=num_clusters,
        eam_sigma_color=eam_sigma_color,
        multi_layer_affinity=multi_layer_affinity,
        differentiable_eigh=differentiable_eigh,
        input_is_imagenet_normalized=True,
    )
    if device is not None:
        model = model.to(device)
    return model
