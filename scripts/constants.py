"""Centralised paths and constants, shared by every approach and benchmark.

All paths are overridable via environment variables so the same code runs on
the cluster, locally, or in CI without edits. Defaults target the CIL
monocular-depth-estimation dataset layout:

    <DATA_ROOT>/train/<basename>_rgb.png      RGB image
    <DATA_ROOT>/train/<basename>_depth.npy    sparse LiDAR depth (metres)
    <DATA_ROOT>/test/<basename>_rgb.png       RGB image (no GT)

Import what you need:

    from scripts.constants import TRAIN_DIR, TEST_DIR, CLEANED_GT_DIR
"""
from __future__ import annotations

import os

# Root of the CIL monocular-depth-estimation dataset.
DATA_ROOT = os.environ.get(
    "CIL_DATA_ROOT", "/cluster/courses/cil/monocular-depth-estimation"
)

# Train / test splits. RGB images and (for train) sparse depth live here, named
# "<basename>_rgb.png" and "<basename>_depth.npy".
TRAIN_DIR = os.environ.get("CIL_TRAIN_DIR", os.path.join(DATA_ROOT, "train"))
TEST_DIR = os.environ.get("CIL_TEST_DIR", os.path.join(DATA_ROOT, "test"))

# File-name conventions for the RGB images and depth maps.
RGB_SUFFIX = "_rgb.png"
DEPTH_SUFFIX = "_depth.npy"

# DROR-cleaned training GT (produced by scripts/clean_gt.py). RGB still loads
# from TRAIN_DIR; only the depth source changes for the cleaned-GT experiments.
CLEANED_GT_DIR = os.environ.get(
    "CIL_CLEANED_GT_DIR", os.path.join(DATA_ROOT, "train_depth_clean")
)

# External benchmarks (NYUv2 / KITTI / ETH3D). One root holds both the raw
# benchmark data and the evaluation outputs written back during scoring.
BENCHMARK_DATA_DIR = os.environ.get("BENCHMARK_DATA_DIR", "data/benchmarks")
BENCHMARK_RESULTS_DIR = os.environ.get("BENCHMARK_RESULTS_DIR", "results/benchmarks")

# ImageNet normalisation stats (encoders are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Native frame size of the dataset and the submission resolution.
NATIVE_SIZE = 560
SUBMISSION_SIZE = 560
