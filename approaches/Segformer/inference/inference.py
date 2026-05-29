#!/usr/bin/env python3
"""Standalone inference for a trained SegFormer-B0 + EAGLE depth model.

Loads a checkpoint (which carries its own training ``args``), rebuilds the
encoder + depth head + EAMs, predicts depth for every test RGB, writes
per-image ``.npy`` predictions at the submission size, and builds the
submission CSV.

Usage (from the repository root):
    python -m approaches.Segformer.inference.inference \
        --ckpt approaches/Segformer/results/<run>/best_checkpoint.pt \
        --test_dir /cluster/courses/cil/monocular-depth-estimation/test \
        --out_dir approaches/Segformer/results/<run>/predictions \
        --out_csv approaches/Segformer/results/<run>/submission.csv
"""
from __future__ import annotations

import argparse
import base64
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD, SUBMISSION_SIZE, TEST_DIR
from approaches.Segformer.model import build_model

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD  = np.array(IMAGENET_STD, dtype=np.float32)


def encode_depth(depth: np.ndarray) -> str:
    """Submission encoding: float16 -> zlib compress -> base64."""
    return base64.b64encode(
        zlib.compress(np.asarray(depth, dtype=np.float16).tobytes(), level=9)
    ).decode("utf-8")


def preprocess_rgb(rgb_path: str, input_size: int, normalize: bool) -> torch.Tensor:
    """Load an RGB image, resize to (input_size, input_size), and tensorise.

    ``normalize=True`` applies ImageNet normalization (SegFormer); otherwise
    the image stays in [0, 1].
    """
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb_f = rgb.astype(np.float32) / 255.0
    if normalize:
        rgb_f = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb_f.transpose(2, 0, 1)).unsqueeze(0)


def load_weights(model, ckpt: dict) -> None:
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.depth_head.load_state_dict(ckpt["depth_head_state_dict"])
    eam_ckpt = ckpt.get("eam_state_dict", {})
    for key, eam in model.eams.items():
        if key in eam_ckpt:
            eam.load_state_dict(eam_ckpt[key])
    if hasattr(model, "affinity_projs"):
        aproj_ckpt = ckpt.get("affinity_proj_state_dict", {})
        for key, proj in model.affinity_projs.items():
            if key in aproj_ckpt:
                proj.load_state_dict(aproj_ckpt[key])


@torch.no_grad()
def run_inference(model, test_dir: str, out_dir: Path, input_size: int,
                  device: torch.device) -> None:
    """Predict depth for every *_rgb.png in test_dir; save as .npy at submission size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(test_dir) if f.endswith("_rgb.png"))
    print(f"inference on {len(files)} test images", flush=True)
    for f in tqdm(files, desc="infer"):
        rgb_t = preprocess_rgb(os.path.join(test_dir, f), input_size,
                               normalize=True).to(device)
        out = model(rgb_t)
        depth = out["depth"]  # (1, H, W)
        depth_t = depth.unsqueeze(1).float()  # (1, 1, H, W)
        depth_sub = F.interpolate(
            depth_t, size=(SUBMISSION_SIZE, SUBMISSION_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()
        base = f.replace("_rgb.png", "")
        np.save(out_dir / f"{base}_pred_depth.npy", depth_sub.astype(np.float32))


def make_submission_csv(pred_dir: Path, csv_out: Path) -> None:
    rows = []
    for p in sorted(pred_dir.glob("*_pred_depth.npy")):
        depth = np.load(p)
        depth = np.nan_to_num(depth, nan=100.0, posinf=100.0, neginf=0.0)
        base = p.stem.replace("_pred_depth", "")
        rows.append({"id": f"{base}_depth", "Depths": encode_depth(depth)})
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "Depths"]).to_csv(csv_out, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_dir", default=TEST_DIR)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--input_size", type=int, default=None,
                   help="Defaults to the checkpoint's training input_size.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    input_size = args.input_size or int(train_args.get("input_size", 256))

    model = build_model(train_args, device).eval()
    load_weights(model, ckpt)

    out_dir = Path(args.out_dir)
    run_inference(model, args.test_dir, out_dir, input_size, device)

    csv_out = Path(args.out_csv) if args.out_csv else out_dir.parent / "submission.csv"
    make_submission_csv(out_dir, csv_out)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
