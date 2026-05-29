"""ResNet34 EAGLE loss wrapper: weighted sum of the four shared loss terms.

The individual terms live in ``scripts.losses`` (shared across approaches).
This wrapper just weights and combines them for the ResNet34 ablation, where
the within-cluster consistency is measured on *linear* predicted depth and the
virtual-normal term converts the linear prediction to log-depth internally.
Pass any weight 0 to disable that term; the EAGLE clustering terms additionally
require EAM logits, so with no active EAMs they are simply absent.

  silog           per-image siRMSE on sparse GT (= leaderboard metric).
  vnl             Yin et al. (ICCV 2019): random-triplet 3D normal agreement.
  eig_cluster     EAGLE Eq. 1: confident, non-degenerate cluster assignment.
  within_cluster  soft-assignment-weighted within-cluster variance of depth.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from scripts.losses import (
    eigen_cluster_loss,
    silog_loss,
    virtual_normal_loss,
    within_cluster_consistency_loss,
)


class EagleDepthLoss(nn.Module):
    def __init__(
        self,
        silog_weight: float          = 1.0,
        eig_cluster_weight: float    = 0.05,
        within_cluster_weight: float = 0.1,
        vnl_weight: float            = 5.0,
    ) -> None:
        super().__init__()
        self.silog_weight          = silog_weight
        self.eig_cluster_weight    = eig_cluster_weight
        self.within_cluster_weight = within_cluster_weight
        self.vnl_weight            = vnl_weight

    def forward(
        self,
        pred_depth: torch.Tensor,                        # [B, H, W]
        gt_depth:   torch.Tensor,                        # [B, H, W]
        logits:     Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = pred_depth.device
        zero   = torch.tensor(0.0, device=device)

        # 1. SILog Loss
        silog = silog_loss(pred_depth.float(), gt_depth.float())

        # 2. Virtual Normal Loss (VNL)
        vnl_val = zero
        pred_4d = pred_depth.unsqueeze(1)  # Reshape to [B, 1, H, W]
        gt_4d   = gt_depth.unsqueeze(1)    # Reshape to [B, 1, H, W]

        if self.vnl_weight > 0:
            # VNL expects log depth, so we safely convert the linear prediction
            student_log = torch.log(pred_4d.clamp(min=1e-3))
            vnl_val = virtual_normal_loss(student_log, gt_4d)

        # 3. EAGLE Clustering Losses
        eig_terms: List[torch.Tensor] = []
        wc_terms:  List[torch.Tensor] = []

        if logits:
            mask = (gt_depth > 1e-3) & torch.isfinite(gt_depth)
            mask_4d = mask.unsqueeze(1)

            for s, lg in logits.items():
                if self.eig_cluster_weight > 0:
                    eig_terms.append(eigen_cluster_loss(lg))
                if self.within_cluster_weight > 0:
                    wc_terms.append(
                        within_cluster_consistency_loss(pred_4d, lg, mask=mask_4d)
                    )

        eig_cluster_val    = (sum(eig_terms) / len(eig_terms)) if eig_terms else zero
        within_cluster_val = (sum(wc_terms) / len(wc_terms))   if wc_terms  else zero

        # 4. Total Weighted Loss
        total = (
            (self.silog_weight * silog)
            + (self.vnl_weight * vnl_val)
            + (self.eig_cluster_weight * eig_cluster_val)
            + (self.within_cluster_weight * within_cluster_val)
        )

        components = {
            "silog":          silog.detach(),
            "vnl":            vnl_val.detach() if isinstance(vnl_val, torch.Tensor) else zero,
            "eig_cluster":    eig_cluster_val.detach() if isinstance(eig_cluster_val, torch.Tensor) else zero,
            "within_cluster": within_cluster_val.detach() if isinstance(within_cluster_val, torch.Tensor) else zero,
        }

        return total, components
