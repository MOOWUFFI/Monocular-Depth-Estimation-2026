"""MobileNet loss wrapper: weighted sum of the four shared loss terms.

The individual terms live in ``scripts.losses`` (shared across approaches).
This wrapper just weights and combines them for the MobileNet ablation, where
the within-cluster consistency is measured on *log*-depth. Set any weight to 0
to disable that term.

  silog        per-image siRMSE on sparse GT (= leaderboard metric).
  eig_cluster  EAGLE Eq. 1: confident, non-degenerate cluster assignment.
  within_cluster  soft-assignment-weighted within-cluster variance of log-depth.
  virtual_normal  Yin et al. 2021: random-triplet 3D normal agreement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.losses import (
    eigen_cluster_loss,
    silog_from_log,
    virtual_normal_loss,
    within_cluster_consistency_loss,
)


class TotalLoss(nn.Module):
    """Weighted sum of the four components. Set weight=0 to disable.

    Multi-stride EAGLE: when the model has EAM at multiple scales, eig_cluster
    and within_cluster are computed at *each* scale and summed (equally
    weighted across scales).
    """

    def __init__(
        self,
        silog_weight: float = 1.0,
        eig_cluster_weight: float = 0.0,
        within_cluster_weight: float = 0.0,
        virtual_normal_weight: float = 0.0,
        virtual_normal_fov_deg: float = 60.0,
        virtual_normal_n_triplets: int = 1024,
    ) -> None:
        super().__init__()
        self.silog_weight = float(silog_weight)
        self.eig_cluster_weight = float(eig_cluster_weight)
        self.within_cluster_weight = float(within_cluster_weight)
        self.virtual_normal_weight = float(virtual_normal_weight)
        self.virtual_normal_fov_deg = float(virtual_normal_fov_deg)
        self.virtual_normal_n_triplets = int(virtual_normal_n_triplets)

    def forward(
        self,
        student_out: dict,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        log_depth = student_out["log_depth"]
        zero = torch.zeros((), device=log_depth.device, dtype=log_depth.dtype)
        comps = {
            "silog": zero.clone(),
            "eig_cluster": zero.clone(),
            "within_cluster": zero.clone(),
            "virtual_normal": zero.clone(),
        }

        total = zero.clone()

        if self.silog_weight > 0:
            l = silog_from_log(log_depth, gt_depth)
            total = total + self.silog_weight * l
            comps["silog"] = l.detach()

        logits_dict = student_out.get("logits", {})
        if logits_dict and self.eig_cluster_weight > 0:
            l = sum(eigen_cluster_loss(lg) for lg in logits_dict.values()) / len(logits_dict)
            total = total + self.eig_cluster_weight * l
            comps["eig_cluster"] = l.detach()

        if logits_dict and self.within_cluster_weight > 0:
            l = sum(
                within_cluster_consistency_loss(log_depth, lg, mask=mask)
                for lg in logits_dict.values()
            ) / len(logits_dict)
            total = total + self.within_cluster_weight * l
            comps["within_cluster"] = l.detach()

        if self.virtual_normal_weight > 0:
            l = virtual_normal_loss(
                log_depth, gt_depth,
                fov_deg=self.virtual_normal_fov_deg,
                n_triplets=self.virtual_normal_n_triplets,
            )
            total = total + self.virtual_normal_weight * l
            comps["virtual_normal"] = l.detach()

        comps["total"] = total.detach()
        return total, comps
