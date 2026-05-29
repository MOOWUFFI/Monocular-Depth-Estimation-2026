#!/usr/bin/env python3
"""Inference + submission CSV for a trained ResNet34 + DepthDecodeHead model.

Loads the saved encoder dir + depth-head file (the checkpoint format written by
``train/train.py``: ``best_encoder/`` + ``best_depth_head.pt``), predicts depth
for every test RGB, writes per-image .npy + magma visualization, and builds the
submission CSV (float16 -> zlib -> base64 per depth map).

Usage (from the repository root):
    python -m approaches.Resnet.inference.inference \\
        --test_dir /cluster/courses/cil/monocular-depth-estimation/test \\
        --output_dir approaches/Resnet/results/predictions \\
        --output_csv submission.csv \\
        --encoder_dir <run>/best_encoder --head_path <run>/best_depth_head.pt \\
        --use_aspp
"""
import argparse
import base64
import glob
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.constants import TEST_DIR
from approaches.Resnet.train.dataset import _IMAGENET_MEAN, _IMAGENET_STD
from approaches.Resnet.model import (
    DepthDecodeHead,
    ResNet34EncoderWrapper,
    _RESNET34_HIDDEN_SIZES,
)


def encode_depth(depth: np.ndarray) -> str:
    """Submission encoding: float16 -> zlib compress -> base64."""
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    return base64.b64encode(compressed).decode("utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Inference for ResNet34 Depth Model")
    p.add_argument("--test_dir", type=str, default=TEST_DIR, help="Folder containing test RGB images")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save predicted depth maps")
    p.add_argument("--output_csv", type=str, default="submission.csv", help="Submission CSV name")
    p.add_argument("--encoder_dir", type=str, required=True, help="Path to best_encoder folder")
    p.add_argument("--head_path", type=str, required=True, help="Path to best_depth_head.pt")
    p.add_argument("--input_size", type=int, default=256, help="Input size used during training")
    p.add_argument("--use_aspp", action="store_true", default=False, help="Enable ASPP in the head (must match training)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"ASPP Enabled: {args.use_aspp}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_csv)

    print("Loading ResNet34 encoder...")
    encoder = ResNet34EncoderWrapper.from_pretrained(args.encoder_dir, pretrained=False).to(device)
    encoder.eval()

    print("Loading Depth Head...")
    depth_head = DepthDecodeHead(
        hidden_sizes=_RESNET34_HIDDEN_SIZES,
        decoder_hidden_size=256,
        use_aspp=args.use_aspp,
    ).to(device)
    depth_head.load_state_dict(torch.load(args.head_path, map_location=device, weights_only=True))
    depth_head.eval()

    image_paths = sorted(glob.glob(os.path.join(args.test_dir, "*_rgb.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.png")) + glob.glob(os.path.join(args.test_dir, "*.jpg")))
    print(f"Found {len(image_paths)} test images.")

    csv_rows = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting Depth"):
            bgr = cv2.imread(img_path)
            orig_h, orig_w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            rgb_resized = cv2.resize(rgb, (args.input_size, args.input_size), interpolation=cv2.INTER_LINEAR)
            rgb_f = rgb_resized.astype(np.float32) / 255.0
            rgb_norm = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
            input_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

            enc_out = encoder(pixel_values=input_tensor)
            pred_depth = depth_head(enc_out.hidden_states, target_size=(args.input_size, args.input_size))
            pred_depth = pred_depth.squeeze(0).squeeze(0)

            pred_depth_resized = F.interpolate(
                pred_depth.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            filename = Path(img_path).stem.replace("_rgb", "")
            idx = filename.split("_")[-1]
            img_id = f"test_{idx}_depth"

            csv_rows.append({"id": img_id, "Depths": encode_depth(pred_depth_resized)})
            np.save(os.path.join(args.output_dir, f"{filename}_depth.npy"), pred_depth_resized)

            depth_normalized = cv2.normalize(pred_depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_magma = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(args.output_dir, f"{filename}_vis.png"), depth_magma)

    print("Building submission file...")
    df = pd.DataFrame(csv_rows, columns=["id", "Depths"])
    df.to_csv(csv_path, index=False)
    print(f"Done! Saved {len(df)} predictions to {csv_path}")


if __name__ == "__main__":
    main()
