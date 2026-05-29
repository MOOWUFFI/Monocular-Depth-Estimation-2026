"""Encode a directory of per-test-image depth .npy files into the submission CSV.

train.py auto-runs the equivalent at end-of-training (unless --no_run_eval).
Standalone usage:

    python -m approaches.Mobilenet.inference.submit_predictions \
        --pred_dir approaches/Mobilenet/results/predictions \
        --out      approaches/Mobilenet/results/submission.csv

Each .npy is expected to be a (560, 560) float depth map. If a prediction is at
a different resolution, it's bilinearly resized to (560, 560). NaN/Inf pixels
are mapped to 100 m (a safe fallback that won't trip the encoder).
"""
from __future__ import annotations

import argparse
import base64
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.constants import SUBMISSION_SIZE


def encode_depth(depth: np.ndarray) -> str:
    return base64.b64encode(
        zlib.compress(np.asarray(depth, dtype=np.float16).tobytes(), level=9)
    ).decode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir).expanduser()
    pred_files = sorted(pred_dir.glob("*_pred_depth.npy"))
    if not pred_files:
        raise SystemExit(f"no *_pred_depth.npy in {pred_dir}")
    print(f"encoding {len(pred_files)} files")

    target = (SUBMISSION_SIZE, SUBMISSION_SIZE)
    rows = []
    for p_path in pred_files:
        depth = np.load(p_path)
        if depth.shape != target:
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                size=target, mode="bilinear", align_corners=False,
            ).squeeze().numpy()
        depth = np.nan_to_num(depth, nan=100.0, posinf=100.0, neginf=0.0)
        base = p_path.stem.replace("_pred_depth", "")
        rows.append({"id": f"{base}_depth", "Depths": encode_depth(depth)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "Depths"]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
