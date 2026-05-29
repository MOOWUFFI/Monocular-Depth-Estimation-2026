"""Render a single .npy depth map as a 2D PNG.

    python -m scripts.visualize_depth depth.npy --out depth.png
    python -m scripts.visualize_depth depth.npy --log  --cmap viridis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npy_path", type=str)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--d_min", type=float, default=0.001)
    p.add_argument("--log", action="store_true")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    depth = np.load(args.npy_path).astype(np.float32)
    invalid = ~np.isfinite(depth) | (depth <= args.d_min)
    masked = np.ma.array(depth, mask=invalid)
    if args.log:
        masked = np.ma.log(np.ma.maximum(masked, args.d_min))

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(masked, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "log(depth)" if args.log else "depth (m)"
    )
    ax.set_title(Path(args.npy_path).name); ax.set_axis_off()
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.npy_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
