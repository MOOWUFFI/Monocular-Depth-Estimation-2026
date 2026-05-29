"""RGB + sparse-depth dataset for the SegFormer-EAGLE approach.

Each sample is ``(rgb_tensor, gt_depth)`` for the training set:
    <data_dir>/<basename>_rgb.png      RGB image
    <data_dir>/<basename>_depth.npy    sparse LiDAR depth (m)

``rgb_tensor`` is float32 CHW; with ``normalize_imagenet=True`` it is
ImageNet-normalised (SegFormer), otherwise it stays in [0, 1] (MobileViT).
``gt_depth`` is in metres (HW).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD  = np.array(IMAGENET_STD, dtype=np.float32)


def _color_jitter(img: np.ndarray) -> np.ndarray:
    """Simple brightness + per-channel contrast jitter."""
    img = img.astype(np.float32)
    img *= 1.0 + random.uniform(-0.20, 0.20)
    for c in range(3):
        mean = img[:, :, c].mean()
        img[:, :, c] = (
            (img[:, :, c] - mean) * (1.0 + random.uniform(-0.20, 0.20)) + mean
        )
    return np.clip(img, 0, 255).astype(np.uint8)


class DepthDataset(Dataset):
    """RGB + ground-truth depth pairs, resized to ``input_size x input_size``.

    Returns ``(rgb_tensor, gt_depth)`` where ``rgb_tensor`` is float32 CHW in
    [0, 1] (MobileViT) or ImageNet-normalised (SegFormer), and ``gt_depth``
    is in metres (HW).  Pass ``normalize_imagenet=False`` for MobileViT.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        input_size: int = 256,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        normalize_imagenet: bool = True,
    ) -> None:
        self.pairs              = pairs
        self.input_size         = input_size
        self.is_train           = is_train
        self.hflip_prob         = hflip_prob
        self.color_jitter_prob  = color_jitter_prob
        self.normalize_imagenet = normalize_imagenet

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.pairs[idx]

        bgr = cv2.imread(rgb_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_depth = np.load(depth_path).astype(np.float32)

        if self.is_train:
            if random.random() < self.hflip_prob:
                rgb      = cv2.flip(rgb, 1)
                gt_depth = cv2.flip(gt_depth, 1)
            if random.random() < self.color_jitter_prob:
                rgb = _color_jitter(rgb)

        sz = (self.input_size, self.input_size)
        if rgb.shape[:2] != sz:
            rgb      = cv2.resize(rgb,      sz, interpolation=cv2.INTER_LINEAR)
            gt_depth = cv2.resize(gt_depth, sz, interpolation=cv2.INTER_NEAREST)

        rgb_f   = rgb.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            rgb_f = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
        rgb_t   = torch.from_numpy(rgb_f.transpose(2, 0, 1))  # (3, H, W)
        depth_t = torch.from_numpy(gt_depth)                  # (H, W)
        return rgb_t, depth_t


def build_data_splits(
    train_dir: str,
    val_split: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Glob RGB/depth pairs under ``train_dir`` and split into train/val."""
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")

    all_rgb = sorted(train_path.glob("*_rgb.png"), key=lambda p: p.name)
    pairs   = [
        (str(rgb), str(rgb).replace("_rgb.png", "_depth.npy"))
        for rgb in all_rgb
        if Path(str(rgb).replace("_rgb.png", "_depth.npy")).exists()
    ]
    if not pairs:
        raise RuntimeError(f"No valid rgb/depth pairs found in {train_path}.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val       = max(1, int(len(pairs) * val_split))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs
