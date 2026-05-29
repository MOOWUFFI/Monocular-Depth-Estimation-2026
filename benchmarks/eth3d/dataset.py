"""ETH3D benchmark dataset/preprocessing helpers.

Self-contained subset used by the ETH3D evaluator: RGB preprocessing for
inference and checkpoint-source detection. No imports outside this package.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# ── ImageNet normalisation constants ──────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_rgb(rgb_path: Path, input_size: int, normalize: bool = True) -> torch.Tensor:
    """Load and preprocess one RGB image for inference.

    Args:
        normalize: If True (default) apply ImageNet mean/std normalisation.
                   Set to False for MobileViT, which expects raw [0, 1] input.

    Returns:
        (1, 3, input_size, input_size) float32 tensor on CPU.
    """
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb_f = rgb.astype(np.float32) / 255.0
    if normalize:
        rgb_f = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb_f.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)


# ── Checkpoint source detection ───────────────────────────────────────────────

def _detect_source(cfg: dict) -> str:
    """Return 'model_train' or 'sf_eagle' based on config keys.

    'model_train' → old ResNet34 checkpoints (always have 'use_aspp' in config).
    'sf_eagle'    → any checkpoint produced by the SegFormer/MobileViT trainer
                    (student_type: segformer | mobilevit).
    """
    if "use_aspp" in cfg:
        return "model_train"
    student = cfg.get("student_type", "resnet34")
    if student == "resnet34":
        return "model_train"
    return "sf_eagle"
