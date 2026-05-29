"""Depth dataset + train/val split for the ResNet34 EAGLE pipeline.

A sample is (rgb, depth): RGB from the dataset image dir, sparse LiDAR depth (m)
loaded as <basename>_depth.npy from `train_dir`. Train-time augmentation is
horizontal flip + brightness/contrast jitter. RGB is ImageNet-normalized by
default to match the pretrained encoder.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD, RGB_SUFFIX, DEPTH_SUFFIX

log = logging.getLogger(__name__)

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

        rgb_f = rgb.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            rgb_f = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
        rgb_t   = torch.from_numpy(rgb_f.transpose(2, 0, 1))  # (3, H, W)
        depth_t = torch.from_numpy(gt_depth)                  # (H, W)
        return rgb_t, depth_t


def build_data_splits(
    rgb_dir: str,
    depth_dir: str,
    val_split: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Build (rgb, depth) pairs and split into train/val.

    RGB images load from `rgb_dir` (named "<basename>_rgb.png"); depth maps
    load from `depth_dir` (named "<basename>_depth.npy"). The two directories
    may differ when training on cleaned GT while keeping the original RGBs.
    """
    rgb_path = Path(rgb_dir)
    depth_path = Path(depth_dir)
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_path}")

    all_rgb = sorted(rgb_path.glob("*" + RGB_SUFFIX), key=lambda p: p.name)
    all_depth = sorted(depth_path.glob("*" + DEPTH_SUFFIX), key=lambda p: p.name)
    print(len(all_rgb))
    print(len(all_depth))

    pairs = [
        (str(rgb), str(rgb_depth))
        for rgb, rgb_depth in zip(all_rgb, all_depth)
    ]
    if not pairs:
        raise RuntimeError(f"No valid rgb/depth pairs found in {depth_path}.")
    log.info("Found %d image-depth pairs.", len(pairs))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val       = max(1, int(len(pairs) * val_split))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    log.info("Split: %d train / %d val", len(train_pairs), len(val_pairs))
    return train_pairs, val_pairs
