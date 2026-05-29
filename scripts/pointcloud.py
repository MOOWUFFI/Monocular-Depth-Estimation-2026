"""Render a depth map + RGB as an interactive 3D point cloud (HTML).

    python -m scripts.pointcloud path/to/depth.npy path/to/rgb.png
    python -m scripts.pointcloud depth.npy rgb.png --out cloud.html --fov 60

Back-projects each valid pixel into camera space using a pinhole model with an
assumed horizontal FOV (the dataset ships no intrinsics). Invalid pixels
(<= d_min, >= d_max, NaN/Inf) are dropped. Writes a standalone HTML you can
open in any browser to orbit/zoom/pan the cloud.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise SystemExit("plotly is required. pip install plotly") from e


def find_matching_rgb(npy_path: Path) -> Path | None:
    stem = npy_path.stem
    for suffix in ("_pred_depth", "_depth"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            for ext in (".png", ".jpg", ".jpeg"):
                cand = npy_path.with_name(f"{base}_rgb{ext}")
                if cand.exists():
                    return cand
    return None


def backproject(depth: np.ndarray, fov_deg: float) -> np.ndarray:
    """Pinhole back-projection. Returns HxWx3 (X right, Y up, Z forward)."""
    H, W = depth.shape
    fx = fy = (W / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (uu - cx) * depth / fx
    Y = -(vv - cy) * depth / fy
    Z = depth.astype(np.float32)
    return np.stack([X, Y, Z], axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npy_path", type=str)
    p.add_argument("rgb_path", type=str, nargs="?", default=None,
                   help="If omitted, auto-detected from the .npy filename.")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--d_min", type=float, default=0.001)
    p.add_argument("--d_max", type=float, default=80.0)
    p.add_argument("--max_points", type=int, default=80000)
    p.add_argument("--marker_size", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    npy_path = Path(args.npy_path)
    rgb_path = Path(args.rgb_path) if args.rgb_path else find_matching_rgb(npy_path)
    if rgb_path is None:
        raise SystemExit(f"could not find RGB partner for {npy_path.name}")

    depth = np.load(npy_path).astype(np.float32)
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    if rgb.shape[:2] != depth.shape:
        rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0]), Image.BILINEAR))

    valid = np.isfinite(depth) & (depth > args.d_min) & (depth < args.d_max)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise SystemExit("no valid depth pixels")

    pts = backproject(depth, args.fov)[valid]
    cols = rgb[valid]
    if len(pts) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pts), size=args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    color_strs = [f"rgb({r},{g},{b})" for r, g, b in cols]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=args.marker_size, color=color_strs, opacity=1.0),
        hovertemplate="x=%{x:.2f}m  y=%{y:.2f}m  z=%{z:.2f}m<extra></extra>",
    )])
    fig.update_layout(
        title=f"{npy_path.name}  ({len(pts):,} of {n_valid:,} valid pts, FOV={args.fov}°)",
        scene=dict(
            xaxis_title="X (m, right)", yaxis_title="Y (m, up)", zaxis_title="Z (m, forward)",
            aspectmode="data",
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=0.0, y=0.0, z=-1.4)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out = Path(args.out) if args.out else npy_path.with_name(npy_path.stem + "_pcd.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"saved {out}  ({len(pts):,} of {n_valid:,} valid pts)")


if __name__ == "__main__":
    main()
