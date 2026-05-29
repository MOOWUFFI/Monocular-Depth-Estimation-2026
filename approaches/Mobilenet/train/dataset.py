"""Sparse-LiDAR depth dataset for monocular depth estimation.

Each sample is (rgb, depth, mask) for the dirty / cleaned training set:
    <data_dir>/<basename>_rgb.png      RGB image
    <data_dir>/<basename>_depth.npy    sparse LiDAR depth (m)

`gt_depth_dir` overrides only the depth source (RGB is still loaded from
`data_dir`); used for the cleaned-GT runs to train on DROR-cleaned GT while
keeping the same RGBs.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD


class SparseDepthDataset(Dataset):
    """Returns (rgb_t, depth_t, mask_t) per sample, all CHW float tensors."""

    def __init__(
        self,
        data_dir: str,
        file_list: list[str],
        is_train: bool = True,
        image_size: int = 576,
        gt_depth_dir: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = file_list
        self.is_train = is_train
        self.image_size = int(image_size)
        self.gt_depth_dir = gt_depth_dir

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        base_name = self.file_list[idx]
        rgb_path = os.path.join(self.data_dir, f"{base_name}_rgb.png")
        depth_dir = self.gt_depth_dir if self.gt_depth_dir else self.data_dir
        depth_path = os.path.join(depth_dir, f"{base_name}_depth.npy")

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.fromarray(np.load(depth_path))

        target = (self.image_size, self.image_size)
        rgb = rgb.resize(target, Image.BILINEAR)
        depth = depth.resize(target, Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                rgb = transforms.functional.hflip(rgb)
                depth = transforms.functional.hflip(depth)
            rgb = self.color_jitter(rgb)

        rgb_t = self.normalize(transforms.functional.to_tensor(rgb))
        depth_t = torch.from_numpy(np.array(depth)).float().unsqueeze(0)
        mask_t = (depth_t >= 1e-3) & (~torch.isnan(depth_t))
        return rgb_t, depth_t, mask_t


def get_train_val_splits(
    data_dir: str, val_count: int = 95
) -> tuple[list[str], list[str]]:
    """First `val_count` images by filename are val; rest are train."""
    all_files = sorted(
        f.replace("_rgb.png", "")
        for f in os.listdir(data_dir)
        if f.endswith("_rgb.png")
    )
    return all_files[val_count:], all_files[:val_count]
