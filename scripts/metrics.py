"""Masked depth metrics. Primary metric is per-image siRMSE — bit-identical
to the leaderboard scorer and to the ``silog_loss`` training term."""
from __future__ import annotations

import torch

LOG_EPS = 1e-6


def _valid(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    v = mask.bool() & (gt > 0) & (pred > 0)
    return pred[v], gt[v]


def sirmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image scale-invariant RMSE.

        siRMSE_i = sqrt(mean(d^2) - mean(d)^2),  d = log(pred_i) - log(gt_i)

    For batched input (B, C, H, W) or (B, H, W), reduces per image
    (images with <100 valid pixels are dropped) and averages.
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred, gt, mask = pred.squeeze(1), gt.squeeze(1), mask.squeeze(1)

    if pred.dim() == 3:
        scores = []
        for b in range(pred.shape[0]):
            v = mask[b].bool() & (gt[b] > 0) & (pred[b] > 0)
            if int(v.sum().item()) < 100:
                continue
            d = pred[b][v].clamp_min(LOG_EPS).log() - gt[b][v].clamp_min(LOG_EPS).log()
            scores.append(torch.sqrt((d.pow(2).mean() - d.mean().pow(2)).clamp_min(0.0)))
        if not scores:
            return torch.tensor(float("nan"), device=pred.device)
        return torch.stack(scores).mean()

    p, g = _valid(pred, gt, mask)
    if p.numel() < 100:
        return torch.tensor(float("nan"), device=pred.device)
    d = p.clamp_min(LOG_EPS).log() - g.clamp_min(LOG_EPS).log()
    return torch.sqrt((d.pow(2).mean() - d.mean().pow(2)).clamp_min(0.0))


def rmse(pred, gt, mask) -> torch.Tensor:
    p, g = _valid(pred, gt, mask)
    if p.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return torch.sqrt(((p - g) ** 2).mean())


def abs_rel(pred, gt, mask) -> torch.Tensor:
    p, g = _valid(pred, gt, mask)
    if p.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return ((p - g).abs() / g).mean()


def delta_thresh(pred, gt, mask, thresh: float) -> torch.Tensor:
    p, g = _valid(pred, gt, mask)
    if p.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    ratio = torch.maximum(p / g, g / p)
    return (ratio < thresh).float().mean()


def all_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    return {
        "sirmse": sirmse(pred, gt, mask),
        "rmse":   rmse(pred, gt, mask),
        "absrel": abs_rel(pred, gt, mask),
        "d1":     delta_thresh(pred, gt, mask, 1.25),
        "d2":     delta_thresh(pred, gt, mask, 1.25 ** 2),
        "d3":     delta_thresh(pred, gt, mask, 1.25 ** 3),
    }
