"""Shared, approach-independent loss building blocks.

These are pure functions used across all four approaches. Each approach's
``train/`` defines its own thin loss *wrapper* that weights and combines these
terms (the wrappers differ — e.g. some operate on log-depth, some on linear
depth), but the underlying maths lives here once.

  silog_loss                     per-image scale-invariant RMSE on sparse GT,
                                 bit-identical to the leaderboard scorer.
  virtual_normal_loss            Yin et al. (ICCV 2019 / TPAMI 2021): random
                                 3-point back-projected triangle-normal
                                 agreement.
  eigen_cluster_loss             EAGLE Eq. 1: confident, non-degenerate cluster
                                 assignment.
  within_cluster_consistency_loss  soft-assignment-weighted within-cluster
                                 variance of a signal (depth / log-depth).
  eicue_argmax                   hard EiCue cluster id per patch (for viz).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def silog_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_val: float = 1e-3,
    sqrt_eps: float = 1e-10,
) -> torch.Tensor:
    """Per-image scale-invariant RMSE on sparse GT (= leaderboard metric).

    ``pred`` and ``gt`` are linear depth, shape [B, H, W]. For each image,
    computes sqrt(mean((d - mean(d))^2) + eps) over its valid pixels
    (d = log(pred) - log(gt)), then averages across images that had >=100
    valid pixels.
    """
    B = pred.shape[0]
    per_image = torch.zeros(B, device=pred.device, dtype=torch.float32)
    valid_count = 0
    for i in range(B):
        valid = (
            (gt[i] > min_val)
            & (pred[i] > min_val)
            & torch.isfinite(gt[i])
            & torch.isfinite(pred[i])
        )
        n = int(valid.sum().item())
        if n < 100:
            continue
        log_diff = torch.log(pred[i][valid]) - torch.log(gt[i][valid])
        centred = log_diff - log_diff.mean()
        per_image[i] = torch.sqrt((centred ** 2).mean() + sqrt_eps)
        valid_count += 1
    if valid_count == 0:
        return pred.sum() * 0.0
    return per_image.sum() / valid_count


def silog_from_log(
    student_log: torch.Tensor,
    gt_depth: torch.Tensor,
    min_val: float = 1e-3,
    sqrt_eps: float = 1e-10,
) -> torch.Tensor:
    """Convenience wrapper: same as ``silog_loss`` but takes predicted
    *log*-depth [B, 1, H, W] and linear GT [B, 1, H, W]."""
    pred = torch.exp(student_log).squeeze(1)
    gt = gt_depth.squeeze(1)
    return silog_loss(pred, gt, min_val=min_val, sqrt_eps=sqrt_eps)


def virtual_normal_loss(
    student_log: torch.Tensor,
    gt_depth: torch.Tensor,
    fov_deg: float = 60.0,
    n_triplets: int = 1024,
    min_dist_3d: float = 0.05,
    min_valid_pixels: int = 100,
) -> torch.Tensor:
    """Virtual normal loss (Yin et al. ICCV 2019 / TPAMI 2021).

    Random valid-pixel triplets are back-projected to 3D under a pinhole
    camera with assumed FOV; the triangle-normal agreement between student
    and GT, |cos(n_pred, n_gt)| (absolute so orientation flips don't matter),
    is penalised. Degenerate triangles are dropped via the 3D edge-norm floor.
    """
    B, _, H, W = student_log.shape
    pred = torch.exp(student_log).squeeze(1)
    gt = gt_depth.squeeze(1)
    fx = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    losses = []
    for b in range(B):
        valid = (
            (gt[b] > 1e-3) & (pred[b] > 1e-3)
            & torch.isfinite(gt[b]) & torch.isfinite(pred[b])
        )
        valid_yx = valid.nonzero(as_tuple=False)
        N = valid_yx.shape[0]
        if N < min_valid_pixels:
            continue

        idx = torch.randint(0, N, (n_triplets, 3), device=pred.device)
        tri = valid_yx[idx]
        y_pix = tri[..., 0].float()
        x_pix = tri[..., 1].float()
        gt_d = gt[b][tri[..., 0], tri[..., 1]]
        pred_d = pred[b][tri[..., 0], tri[..., 1]]

        X_gt = (x_pix - cx) * gt_d / fx
        Y_gt = (y_pix - cy) * gt_d / fx
        X_pred = (x_pix - cx) * pred_d / fx
        Y_pred = (y_pix - cy) * pred_d / fx
        P_gt = torch.stack([X_gt, Y_gt, gt_d], dim=-1)
        P_pred = torch.stack([X_pred, Y_pred, pred_d], dim=-1)

        n_gt = torch.cross(P_gt[:, 1] - P_gt[:, 0], P_gt[:, 2] - P_gt[:, 0], dim=-1)
        n_pred = torch.cross(P_pred[:, 1] - P_pred[:, 0], P_pred[:, 2] - P_pred[:, 0], dim=-1)
        norm_gt = n_gt.norm(dim=-1)
        norm_pred = n_pred.norm(dim=-1)
        keep = (norm_gt > min_dist_3d) & (norm_pred > min_dist_3d)
        if int(keep.sum().item()) == 0:
            continue

        n_gt_u = n_gt / norm_gt.unsqueeze(-1).clamp_min(1e-8)
        n_pred_u = n_pred / norm_pred.unsqueeze(-1).clamp_min(1e-8)
        cos_sim = (n_gt_u * n_pred_u).sum(dim=-1).abs()
        losses.append((1.0 - cos_sim[keep]).mean())

    if not losses:
        return torch.zeros((), device=student_log.device, dtype=student_log.dtype)
    return torch.stack(losses).mean()


def eigen_cluster_loss(logits: torch.Tensor) -> torch.Tensor:
    """EAGLE Eq. 1: L_eig = -(1/N) sum_i psi_i . P_i,  psi = softmax(P).

    Pulls each token toward a confident, non-degenerate cluster assignment and
    trains the cluster centers jointly.
    """
    psi = F.softmax(logits, dim=-1)
    return -(psi * logits).sum(dim=-1).mean()


def within_cluster_consistency_loss(
    signal: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-assignment-weighted within-cluster variance of ``signal``.

    signal: [B, 1, H, W]  full-res signal (predicted depth, log-depth, ...)
    logits: [B, N, C]     EAM pre-softmax cluster scores at low res
    mask:   [B, 1, H, W]  optional bool validity (max-pooled to the EAM grid)
    """
    B, _, H, W = signal.shape
    N = logits.shape[1]
    h = w = int(round(math.sqrt(N)))
    assert h * w == N, f"expected square EAM grid, got N={N}"

    s_low = F.adaptive_avg_pool2d(signal, output_size=(h, w))
    s_flat = s_low.reshape(B, N, 1)
    if mask is not None:
        m_low = F.adaptive_max_pool2d(mask.float(), output_size=(h, w))
        m_flat = m_low.reshape(B, N, 1)
    else:
        m_flat = torch.ones_like(s_flat)

    assign = F.softmax(logits, dim=-1)
    weights = assign * m_flat
    cluster_w = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    cluster_mean = (weights * s_flat).sum(dim=1, keepdim=True) / cluster_w
    cluster_var = (weights * (s_flat - cluster_mean).pow(2)).sum(dim=1) / cluster_w.squeeze(1)
    return cluster_var.mean()


def eicue_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Hard EiCue cluster id per patch (EAGLE Eq. 2, for visualisation)."""
    return logits.argmax(dim=-1)


# Backwards-friendly alias: some code refers to the per-image siRMSE term as
# ``silog``.
silog = silog_loss
