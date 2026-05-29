"""Bulk-denoise every train-set GT depth map in parallel.

For each ``<train_dir>/*_depth.npy``, runs DROR (Charron 2018) + a near-camera
Z-clip to drop LiDAR self-contamination, and saves the cleaned depth to
``<out_dir>/<basename>_depth.npy``. Removed pixels become 0.0 (so they fall
out of the validity mask downstream). This produces the cleaned GT consumed by
the cleaned-GT experiments (``base_aspp`` and later).

Usage:
    python -m scripts.clean_gt \\
        --train_dir /cluster/courses/cil/monocular-depth-estimation/train \\
        --out_dir   ./train_depth_clean \\
        --workers 8

Defaults pull ``--train_dir`` / ``--out_dir`` from scripts.constants
(TRAIN_DIR / CLEANED_GT_DIR). Wall time on a ~22k-image train set:
~2-3 min on 8 cores.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.constants import CLEANED_GT_DIR, TRAIN_DIR


def _save_atomic(path, arr):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, str(path))


def _backproject(depth, fov_deg):
    H, W = depth.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (uu - cx) * depth / fx
    Y = -(vv - cy) * depth / fx
    Z = depth.astype(np.float32)
    return np.stack([X, Y, Z], axis=-1)


def _dror_mask(pts, alpha=0.02, beta=0.05, k_min=20):
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(pts)
    ranges = np.linalg.norm(pts, axis=-1)
    radii = np.maximum(alpha * ranges, beta)
    k = min(k_min + 1, len(pts))
    dists, _ = tree.query(pts, k=k, workers=1)
    if k <= k_min:
        return np.zeros(len(pts), dtype=bool)
    return dists[:, k_min] <= radii


def _denoise_one(args):
    src, dst, fov, d_min, d_max, alpha, beta, k_min, near_clip = args
    depth = np.load(src).astype(np.float32)
    valid_2d = np.isfinite(depth) & (depth > d_min) & (depth < d_max)
    if not valid_2d.any():
        _save_atomic(dst, depth)
        return src, 0, 0

    flat_valid = np.flatnonzero(valid_2d.reshape(-1))
    pts3 = _backproject(depth, fov).reshape(-1, 3)[flat_valid]
    keep = _dror_mask(pts3, alpha=alpha, beta=beta, k_min=k_min)
    if near_clip > 0:
        keep &= pts3[:, 2] >= near_clip

    cleaned = depth.copy()
    cleaned.reshape(-1)[flat_valid[~keep]] = 0.0
    _save_atomic(dst, cleaned)
    return src, int(keep.sum()), int(len(keep))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default=TRAIN_DIR)
    p.add_argument("--out_dir", default=CLEANED_GT_DIR)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--d_min", type=float, default=0.001)
    p.add_argument("--d_max", type=float, default=80.0)
    p.add_argument("--dror_alpha", type=float, default=0.02)
    p.add_argument("--dror_beta", type=float, default=0.05)
    p.add_argument("--dror_k_min", type=int, default=20)
    p.add_argument("--near_clip", type=float, default=0.5)
    p.add_argument("--skip_existing", action="store_true", default=True)
    args = p.parse_args()

    src_dir = Path(args.train_dir)
    dst_dir = Path(args.out_dir); dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*_depth.npy"))
    if not files:
        raise SystemExit(f"no *_depth.npy in {src_dir}")

    jobs = []
    for src in files:
        dst = dst_dir / src.name
        if args.skip_existing and dst.exists():
            continue
        jobs.append((str(src), str(dst),
                     args.fov, args.d_min, args.d_max,
                     args.dror_alpha, args.dror_beta, args.dror_k_min, args.near_clip))
    print(f"==> {len(jobs)} files to process ({len(files) - len(jobs)} skipped, "
          f"{args.workers} workers)")

    t0 = time.time()
    total_kept = total_in = 0
    with mp.Pool(args.workers) as pool:
        for i, (_, kept, n) in enumerate(pool.imap_unordered(_denoise_one, jobs), 1):
            total_kept += kept; total_in += n
            if i % 200 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                pct = 100 * total_kept / max(total_in, 1)
                print(f"  [{i:5d}/{len(jobs)}] {pct:.1f}% kept  "
                      f"rate={rate:.1f}/s  ETA={eta:.0f}s")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
