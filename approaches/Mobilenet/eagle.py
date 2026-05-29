"""EAGLE: Eigen Aggregation module (Kim et al., CVPR 2024) adapted for depth.

Builds an affinity W = A_color + A_seg at one spatial scale, computes the
symmetric-normalised Laplacian, takes the k smallest non-trivial eigenvectors,
and produces soft cluster assignments via learnable cluster centers. The
outputs feed two auxiliary losses (``eigen_cluster_loss`` and
``within_cluster_consistency_loss``, both in ``scripts.losses``) — they do NOT
modify the decoder's feature path.

Multi-stride: just instantiate one EigenAggregationModule per scale, each sized
to the channel count at that decoder stage. The module is stride-agnostic —
feed it a feature map and the corresponding RGB downsampled to that grid.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def color_affinity(rgb_low: torch.Tensor, sigma_color: float = 0.2) -> torch.Tensor:
    """Gaussian RBF color affinity over every patch pair. rgb_low in [0, 1]."""
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
    d = W.sum(dim=-1).clamp_min(1e-6)
    d_inv_sqrt = d.rsqrt()
    W_norm = W * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1)
    eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype).expand_as(W)
    return 0.5 * ((eye - W_norm) + (eye - W_norm).transpose(-1, -2))


class EigenAggregationModule(nn.Module):
    """One scale of EAGLE EiCue.

    Forward returns ``(feat, U, eigvals, logits)``:
        feat:    untouched input — decoder consumes this
        U:       [B, N, k]   leading non-trivial eigenvectors of L_sym
        eigvals: [B, k]
        logits:  [B, N, C]   pre-softmax cluster scores; feeds the losses
    """

    def __init__(
        self,
        channels: int,
        k: int = 4,
        num_clusters: int = 10,
        sigma_color: float = 0.2,
        drop_first: int = 1,
        eigh_eps_reg: float = 1e-2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.k = k
        self.num_clusters = num_clusters
        self.sigma_color = sigma_color
        self.drop_first = drop_first
        self.eigh_eps_reg = float(eigh_eps_reg)

        self.proj_in = nn.Conv2d(channels, channels, 1, bias=False)
        self.cluster_centers = nn.Parameter(
            torch.randn(k, num_clusters) * (1.0 / math.sqrt(k))
        )

    def _eigh_safe(self, L: torch.Tensor):
        """Always-fp32 eigh with jitter-retry. eigh backward isn't needed
        (we detach the eigvecs that feed cluster_centers); we still want
        forward to never NaN."""
        orig_dtype = L.dtype
        with torch.amp.autocast(device_type="cuda", enabled=False):
            L32 = L.float()
            try:
                eigvals, eigvecs = torch.linalg.eigh(L32)
            except Exception:
                eye = torch.eye(L32.shape[-1], device=L32.device, dtype=L32.dtype).expand_as(L32)
                eigvals, eigvecs = torch.linalg.eigh(L32 + 1e-2 * eye)
        return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)

    def forward(self, feat: torch.Tensor, rgb_low: torch.Tensor):
        """feat: [B, C, h, w] aligned with rgb_low: [B, 3, h, w] in [0,1]."""
        B, C, h, w = feat.shape
        N = h * w

        K_sem = semantic_affinity(self.proj_in(feat)).clamp_min(0.0)
        K_col = color_affinity(rgb_low, self.sigma_color).to(K_sem.dtype)
        W = K_sem + K_col

        L = symmetric_normalized_laplacian(W)
        eigvals_full, eigvecs_full = self._eigh_safe(L)

        start = self.drop_first
        end = start + self.k
        U = eigvecs_full[:, :, start:end].detach()
        eigvals = eigvals_full[:, start:end].detach()

        logits = U @ self.cluster_centers.to(U.dtype)
        return feat, U, eigvals, logits
