"""Linear warmup + polynomial decay scheduler.

Warmup lets the freshly-initialised head ramp up gently before the pretrained
encoder LR (already at encoder_mult * base_lr) starts compounding gradient
noise into the backbone.
"""
from __future__ import annotations

import torch


def warmup_poly(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    power: float = 0.9,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if step >= total_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return (1.0 - progress) ** power
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
