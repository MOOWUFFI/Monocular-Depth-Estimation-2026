"""Evaluate the MobileNet checkpoints on the KITTI Eigen test split.

Cross-dataset generalisation check. Uses Marigold's curated split:
    https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti/

Protocol (standard monocular-depth eval, Eigen / Garg / Lee lineage):
    1. Resize each KITTI RGB to the model's training resolution (576x576).
    2. Predict depth, resize prediction back to original (H, W).
    3. Apply the Garg crop (central image region with dense LiDAR GT).
    4. Per-image median scaling: pred *= median(gt_valid) / median(pred_valid).
    5. Clip pred to [1e-3, 80] m.
    6. Compute silog / absrel / d1 over valid pixels; average across images.

Caveat: the course-dataset -> KITTI domain gap is large. Expect weak numbers
across the board — this is generalisation, not in-distribution accuracy.

Usage (from the repository root):
    # one-time data download
    wget -P data/benchmarks/kitti \\
        https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti/kitti_eigen_split_test.tar
    tar -xf data/benchmarks/kitti/kitti_eigen_split_test.tar -C data/benchmarks/kitti/

    # run the eval (after the MobileNet experiments have been trained)
    python -m benchmarks.kitti.inference \\
        --data_root data/benchmarks/kitti \\
        --ckpt_root approaches/Mobilenet/results
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from scripts.constants import BENCHMARK_DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
from approaches.Mobilenet.model import build_model


# ---------------------------------------------------------------------------
# Eigen / Garg eval protocol helpers
# ---------------------------------------------------------------------------

def garg_crop_mask(h: int, w: int) -> np.ndarray:
    """Garg 2016 crop: keep the central rectangle where KITTI LiDAR is dense."""
    mask = np.zeros((h, w), dtype=bool)
    y_top = int(0.40810811 * h)
    y_bot = int(0.99189189 * h)
    x_left = int(0.03594771 * w)
    x_right = int(0.96405229 * w)
    mask[y_top:y_bot, x_left:x_right] = True
    return mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Returns (silog, absrel, d1) on valid pixels, or None if too few.

    silog:  per-image scale-invariant RMSE — sqrt(mean(d^2) - mean(d)^2)
            with d = log(pred) - log(gt). Matches the training loss formula
            and the KITTI official depth benchmark.
    absrel: mean |pred - gt| / gt
    d1:     fraction of pixels with max(pred/gt, gt/pred) < 1.25
    """
    valid = mask & (gt > 1e-3) & (pred > 1e-3) & np.isfinite(gt) & np.isfinite(pred)
    if int(valid.sum()) < 100:
        return None
    p, g = pred[valid], gt[valid]
    log_diff = np.log(p) - np.log(g)
    silog = float(np.sqrt(max(0.0, np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)))
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    return silog, absrel, d1


# ---------------------------------------------------------------------------
# Model + data helpers
# ---------------------------------------------------------------------------

def _load_experiment_args(proj_dir: Path, ckpt: dict) -> dict:
    """Reconstruct the training args for an experiment.

    Prefer the sidecar args.json written by train.py; fall back to the ``args``
    dict embedded in the checkpoint (so a standalone checkpoint also works).
    The encoder is never re-initialised from ImageNet here — the loaded
    state_dict overwrites it — so pretrained_encoder is forced off to avoid a
    needless download.
    """
    args_path = proj_dir / "args.json"
    if args_path.exists():
        with open(args_path) as f:
            args_dict = json.load(f)
    else:
        args_dict = dict(ckpt.get("args", {}))
    args_dict["pretrained_encoder"] = False
    return args_dict


def load_kitti_depth(depth_path: Path) -> np.ndarray:
    """KITTI depth: 16-bit PNG, depth_meters = pixel / 256.0. 0 = invalid."""
    return np.array(Image.open(depth_path)).astype(np.float32) / 256.0


def find_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    """Discover (rgb, depth) pairs in Marigold's KITTI Eigen split.

    Marigold's layout splits images and depth into two parallel trees:
        <root>/<date>/<drive>/image_02/data/<frame>.png                  <- RGB
        <root>/<drive>/proj_depth/groundtruth/image_02/<frame>.png       <- depth
    (drive nests under date for RGB, drive is flat at root for depth.)

    Falls back to a filelist or flat rgb/depth dirs if that fails.
    """
    pairs: list[tuple[Path, Path]] = []

    # Primary: Marigold's split-tree layout
    for rgb in sorted(data_root.rglob("image_02/data/*.png")):
        try:
            rel = rgb.relative_to(data_root).parts
        except ValueError:
            continue
        # expect ('<date>', '<drive>', 'image_02', 'data', '<frame>.png')
        if len(rel) != 5 or rel[2] != "image_02" or rel[3] != "data":
            continue
        drive, frame = rel[1], rel[-1]
        depth = data_root / drive / "proj_depth" / "groundtruth" / "image_02" / frame
        if depth.exists():
            pairs.append((rgb, depth))
    if pairs:
        return pairs

    # Fallback: filelist of "rgb_path depth_path" lines
    for name in ("eigen_test_files_with_gt.txt", "eigen_test_files.txt", "filelist.txt"):
        fl = data_root / name
        if fl.exists():
            for line in fl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    rgb_p = (data_root / parts[0]).resolve()
                    dep_p = (data_root / parts[1]).resolve()
                    if rgb_p.exists() and dep_p.exists():
                        pairs.append((rgb_p, dep_p))
            if pairs:
                return pairs

    # Fallback: flat rgb/ + depth/ dirs with matched filenames
    rgb_dir = data_root / "rgb"
    dep_dir = data_root / "depth"
    if rgb_dir.is_dir() and dep_dir.is_dir():
        for rgb in sorted(rgb_dir.glob("*.png")):
            depth = dep_dir / rgb.name
            if depth.exists():
                pairs.append((rgb, depth))
    return pairs


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_experiment(
    proj_dir: Path,
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    train_image_size: int = 576,
    median_scale: bool = True,
    garg_crop: bool = True,
    max_depth: float = 80.0,
) -> tuple[float, float, float, int]:
    ckpt_path = proj_dir / "checkpoints" / "best.pth"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = _load_experiment_args(proj_dir, ck)
    model = build_model(args_dict, device).eval()
    model.load_state_dict(ck["model_state_dict"])
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    results = []
    for i, (rgb_path, depth_path) in enumerate(pairs):
        rgb = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = rgb.size
        gt = load_kitti_depth(depth_path)

        rgb_resized = rgb.resize((train_image_size, train_image_size), Image.BILINEAR)
        rgb_t = norm(transforms.functional.to_tensor(rgb_resized)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(rgb_t)
        pred_low = torch.exp(out["log_depth"]).squeeze().float().cpu().numpy()

        # Resize prediction back to GT shape
        pred = np.array(Image.fromarray(pred_low).resize((orig_w, orig_h), Image.BILINEAR))

        valid_gt = (gt > 1e-3) & np.isfinite(gt) & (gt < max_depth)
        if garg_crop:
            valid_gt &= garg_crop_mask(*gt.shape)

        if median_scale and int(valid_gt.sum()) >= 100:
            scale = float(np.median(gt[valid_gt])) / max(float(np.median(pred[valid_gt])), 1e-6)
            pred = pred * scale

        pred = np.clip(pred, 1e-3, max_depth)
        m = compute_metrics(pred, gt, valid_gt)
        if m is not None:
            results.append(m)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}]", flush=True)

    if not results:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean([m[0] for m in results])),
        float(np.mean([m[1] for m in results])),
        float(np.mean([m[2] for m in results])),
        len(results),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(BENCHMARK_DATA_DIR, "kitti"),
                   help="Unpacked Marigold kitti_eigen_split_test/ directory.")
    p.add_argument("--ckpt_root", default="approaches/Mobilenet/results",
                   help="Directory containing the experiment subdirs.")
    p.add_argument("--experiments", nargs="+",
                   default=["scratch", "base", "base_aspp",
                            "base_aspp_eagle", "base_aspp_eagle_vn"])
    p.add_argument("--train_image_size", type=int, default=576)
    p.add_argument("--no_median_scale", action="store_true")
    p.add_argument("--no_garg_crop", action="store_true")
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--out_json", type=str, default=None,
                   help="Optional path to dump the metrics dict as JSON.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    pairs = find_pairs(Path(args.data_root).expanduser())
    if not pairs:
        raise SystemExit(
            f"no rgb-depth pairs found under {args.data_root}. "
            f"Expected Marigold KITTI-raw layout — check the unpacked dir."
        )
    print(f"found {len(pairs)} test pairs under {args.data_root}", flush=True)

    print(f"\n{'experiment':<25} {'silog':>10} {'absrel':>10} {'d1':>10}  N")
    print("-" * 65)
    out: dict = {"pairs": len(pairs), "experiments": {}}
    for exp in args.experiments:
        proj = Path(args.ckpt_root) / exp
        if not (proj / "checkpoints" / "best.pth").exists():
            print(f"{exp:<25} -- best.pth not found at {proj}")
            continue
        print(f"== {exp} ==", flush=True)
        silog, absrel, d1, n = evaluate_experiment(
            proj, pairs, device,
            train_image_size=args.train_image_size,
            median_scale=not args.no_median_scale,
            garg_crop=not args.no_garg_crop,
            max_depth=args.max_depth,
        )
        print(f"{exp:<25} {silog:>10.4f} {absrel:>10.4f} {d1:>10.4f}  {n}",
              flush=True)
        out["experiments"][exp] = {
            "silog": silog, "absrel": absrel, "d1": d1, "n": n,
        }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
