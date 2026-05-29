"""Model-side EAGLE: the Eigen Aggregation Module and its affinity / Laplacian
helpers (Kim et al., "EAGLE: Eigen Aggregation Learning for Object-Centric
Unsupervised Semantic Segmentation", CVPR 2024), adapted for depth estimation.

This module builds an affinity ``A = A_color + A_seg`` at one spatial scale,
computes the symmetric-normalised graph Laplacian, takes the k smallest
non-trivial eigenvectors, and produces soft cluster-assignment logits via a set
of learnable cluster centers.

The module is a *side channel* only: it returns ``(feat, U, eigvals, logits)``
where ``feat`` is the unmodified input (when ``use_residual=False``). The
eigenvectors and logits feed the auxiliary losses (``eigen_cluster_loss`` and
``within_cluster_consistency_loss`` in ``scripts.losses``); they do NOT modify
the feature maps fed to the depth decoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Affinity helpers
# ---------------------------------------------------------------------------

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
    """A_seg = S S^T on L2-normalised features (paper §3.2.1 II)."""
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


# ---------------------------------------------------------------------------
# Eigen Aggregation Module (paper-faithful: cluster map only, no feature
# aggregation into the decoder path)
# ---------------------------------------------------------------------------

class EigenAggregationModule(nn.Module):
    """Compute EiCue (eigenvector-derived cluster assignment) at one spatial scale.

    Returns three things that feed only into the loss, never into the
    decoder. The decoder consumes the unmodified `feat` as before.
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

        # proj_in transforms `feat` into the space used for semantic
        # affinity — the paper's S = S_θ(K) role. Kept tiny (1x1 conv,
        # no bias) so we don't blow up param count.
        self.proj_in = nn.Conv2d(channels, channels, 1, bias=False)

        # Cluster centers C ∈ R^{k × num_clusters}, learned end-to-end via
        # L_eig (paper Eq. 1). Small init so the initial softmax is near-uniform.
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
                # Add an order-of-magnitude more jitter and try again. If
                # this still fails, propagate — at least we won't silently
                # corrupt the run.
                jitter = max(self.eigh_eps_reg * 10.0, 1e-2)
                eye = torch.eye(
                    L32.shape[-1], device=L32.device, dtype=L32.dtype
                ).expand_as(L32)
                eigvals, eigvecs = torch.linalg.eigh(L32 + jitter * eye)
        if self.differentiable_eigh:
            # When the backward through eigh produces NaN/Inf (close
            # eigenvalues → 1/(λ_i − λ_j) blowup), zero those gradients so
            # the optimizer step doesn't propagate corruption into the
            # encoder. Without this guard the second batch is already
            # poisoned. With it, we degrade gracefully to "no spectral-path
            # gradient on this batch" instead of crashing the whole run.
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
        the SEMANTIC affinity. Defaults to `feat` if not provided. Pass
        a multi-layer concat-projected tensor here to get the paper's
        last-3-layer-concat behaviour.
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
            # Residual uses ORIGINAL feat (not affinity_feat) so the decoder
            # input shape stays at the bottleneck channels.
            f_flat = feat.reshape(B, C, N)
            coeffs = torch.einsum("bnk,bcn->bck", U, f_flat)
            f_eig_flat = torch.einsum("bnk,bck->bcn", U, coeffs)
            f_eig = f_eig_flat.reshape(B, C, h, w)
            out_feat = feat + self.proj_out(f_eig)
        else:
            out_feat = feat

        return out_feat, U, eigvals, logits
