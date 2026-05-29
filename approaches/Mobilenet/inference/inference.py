"""Standalone inference for a trained MobileNet depth model.

Loads a checkpoint (which carries its own training ``args``), predicts depth for
every test RGB, writes per-image ``.npy`` predictions at (560, 560), and builds
the submission CSV.

Usage (from the repository root):
    python -m approaches.Mobilenet.inference.inference \
        --ckpt approaches/Mobilenet/results/checkpoints/best.pth \
        --test_dir /cluster/courses/cil/monocular-depth-estimation/test \
        --out_dir approaches/Mobilenet/results/predictions \
        --out_csv approaches/Mobilenet/results/submission.csv
"""
from __future__ import annotations

import argparse
import base64
import os
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD, SUBMISSION_SIZE, TEST_DIR
from approaches.Mobilenet.model import build_model


def encode_depth(depth: np.ndarray) -> str:
    """Submission encoding: float16 -> zlib compress -> base64."""
    return base64.b64encode(
        zlib.compress(np.asarray(depth, dtype=np.float16).tobytes(), level=9)
    ).decode("utf-8")


@torch.no_grad()
def run_inference(model, test_dir: str, out_dir: Path, image_size: int,
                  device: torch.device) -> None:
    """Predict depth for every *_rgb.png in test_dir; save as .npy at submission size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(test_dir) if f.endswith("_rgb.png"))
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    print(f"inference on {len(files)} test images", flush=True)
    for f in tqdm(files, desc="infer"):
        img = Image.open(os.path.join(test_dir, f)).convert("RGB")
        img_t = norm(
            transforms.functional.to_tensor(img.resize((image_size, image_size), Image.BILINEAR))
        ).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda"):
            out = model(img_t)
        depth = torch.exp(out["log_depth"]).squeeze().float().cpu().numpy()
        depth_t = torch.from_numpy(depth)[None, None]
        depth_sub = F.interpolate(depth_t, size=(SUBMISSION_SIZE, SUBMISSION_SIZE),
                                  mode="bilinear", align_corners=False).squeeze().numpy()
        base = f.replace("_rgb.png", "")
        np.save(out_dir / f"{base}_pred_depth.npy", depth_sub.astype(np.float32))


def make_submission_csv(pred_dir: Path, csv_out: Path) -> None:
    import pandas as pd

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
    p.add_argument("--image_size", type=int, default=None,
                   help="Defaults to the checkpoint's training image_size.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ck.get("args", {})
    image_size = args.image_size or int(train_args.get("image_size", 576))
    model = build_model(train_args, device).eval()
    model.load_state_dict(ck["model_state_dict"])

    out_dir = Path(args.out_dir)
    run_inference(model, args.test_dir, out_dir, image_size, device)

    csv_out = Path(args.out_csv) if args.out_csv else out_dir.parent / "submission.csv"
    make_submission_csv(out_dir, csv_out)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
