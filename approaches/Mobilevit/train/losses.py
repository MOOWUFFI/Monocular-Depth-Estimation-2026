"""EAGLE depth loss wrapper for the MobileViT-EAGLE approach.

Composite loss = silog_weight * SILog
               + eig_cluster_weight * L_eig
               + within_cluster_weight * L_within_cluster

The individual terms live in ``scripts.losses`` (shared across approaches).
This wrapper just weights and combines them. The ``within_cluster`` term
operates on the predicted *metric* depth (not log-depth) at the EAM grid
resolution, which is appropriate for the metric-depth head used here.

  silog            scale-invariant log loss on metric depth prediction.
  eig_cluster      EAGLE Eq. 1: confident, non-degenerate cluster assignment.
  within_cluster   soft-assignment-weighted within-cluster variance of depth.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from scripts.losses import (
    eigen_cluster_loss,
    silog_loss,
    within_cluster_consistency_loss,
)


class EagleDepthLoss(nn.Module):
    """Composite EAGLE depth loss (weighted sum of the three terms).

    EAGLE terms are computed at *each* active EAM stage and averaged across
    stages. Set any weight to 0 to disable that term.
    """

    def __init__(
        self,
        silog_weight: float          = 1.0,
        eig_cluster_weight: float     = 0.05,
        within_cluster_weight: float  = 0.1,
    ) -> None:
        super().__init__()
        self.silog_weight          = silog_weight
        self.eig_cluster_weight    = eig_cluster_weight
        self.within_cluster_weight = within_cluster_weight

    def forward(
        self,
        pred_depth: torch.Tensor,             # (B, H, W)
        gt_depth:   torch.Tensor,             # (B, H, W)
        logits:     Dict[int, torch.Tensor],  # stage -> (B, N, K)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = pred_depth.device
        zero   = torch.tensor(0.0, device=device)

        # -- SILog -----------------------------------------------------------
        silog = silog_loss(pred_depth.float(), gt_depth.float())

        # -- EAGLE terms (averaged over active EAM stages) -------------------
        eig_terms: List[torch.Tensor] = []
        wc_terms:  List[torch.Tensor] = []

        if logits:
            # Build a validity mask from GT depth (for within-cluster loss)
            mask = (gt_depth > 1e-3) & torch.isfinite(gt_depth)  # (B, H, W)
            mask_4d = mask.unsqueeze(1)                            # (B, 1, H, W)
            pred_4d = pred_depth.unsqueeze(1)                      # (B, 1, H, W)

            for s, lg in logits.items():
                if self.eig_cluster_weight > 0:
                    eig_terms.append(eigen_cluster_loss(lg))
                if self.within_cluster_weight > 0:
                    wc_terms.append(
                        within_cluster_consistency_loss(pred_4d, lg, mask=mask_4d)
                    )

        eig_cluster_val    = (sum(eig_terms) / len(eig_terms)) if eig_terms else zero
        within_cluster_val = (sum(wc_terms)  / len(wc_terms))  if wc_terms  else zero

        total = (
            self.silog_weight          * silog
            + self.eig_cluster_weight    * eig_cluster_val
            + self.within_cluster_weight * within_cluster_val
        )

        components = {
            "silog":          silog.detach(),
            "eig_cluster":    eig_cluster_val.detach()    if isinstance(eig_cluster_val,    torch.Tensor) else zero,
            "within_cluster": within_cluster_val.detach() if isinstance(within_cluster_val, torch.Tensor) else zero,
        }
        return total, components
