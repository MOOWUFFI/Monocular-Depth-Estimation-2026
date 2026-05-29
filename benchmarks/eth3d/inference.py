#!/usr/bin/env python3
"""Inference + depth evaluation on the ETH3D benchmark dataset.

Loads the best checkpoint from one or more checkpoint dirs, runs it on every
bench_XXXXXX_rgb.png / bench_XXXXXX_depth.npy pair in the data dir, and reports
three metrics used in the Depth Anything V2 paper:
  - SI-RMSE  : scale-invariant RMSE (log-space, optimal multiplicative shift)
  - AbsRel   : mean absolute relative error after the same optimal scale alignment
  - delta1   : % of pixels where max(pred/gt, gt/pred) < 1.25 (after alignment)

This script is self-contained: every architecture is defined in
``benchmarks.eth3d.models`` so it can load any of the project's checkpoints
without importing the training packages.

Run as a module:

    python -m benchmarks.eth3d.inference \\
        --data_dir        data/benchmarks/eth3d \\
        --checkpoint_dirs data/benchmarks/models
"""

import argparse
import json
import os
from pathlib import Path

import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from transformers import SegformerModel, SegformerForSemanticSegmentation, MobileViTModel

from benchmarks.eth3d.models import (
    build_baseline_model,
    EAGLEDepthModel,
    DepthDecodeHead,
    ResNet34EncoderWrapper,
    ResNet34DepthDecodeHead,
    TinyDepthUNet,
    _RESNET34_HIDDEN_SIZES,
)
from benchmarks.eth3d.dataset import preprocess_rgb, _detect_source
from scripts.constants import BENCHMARK_DATA_DIR

# Benchmark data lives under BENCHMARK_DATA_DIR/eth3d by default; the trained
# checkpoints default to BENCHMARK_DATA_DIR/models. Both are overridable on the
# command line (see parse_args).
_DEFAULT_DATA_DIR  = os.path.join(BENCHMARK_DATA_DIR, "eth3d")
_DEFAULT_CKPT_DIR  = os.path.join(BENCHMARK_DATA_DIR, "models")


# ── Metrics ───────────────────────────────────────────────────────────────────

def _valid_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return (gt > 1e-3) & (pred > 1e-3) & np.isfinite(gt) & np.isfinite(pred)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Return SI-RMSE, AbsRel, and delta1 for a single prediction/GT pair.

    SI-RMSE uses the optimal log-space multiplicative shift (alpha).
    AbsRel and delta1 use least-squares affine alignment (scale + shift),
    matching the Depth Anything V2 evaluation protocol exactly.
    delta1 is reported as a fraction in [0, 1].
    """
    valid = _valid_mask(pred, gt)
    if valid.sum() == 0:
        return {"si_rmse": float("nan"), "absrel": float("nan"), "delta1": float("nan")}

    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)

    log_delta = np.log(p) - np.log(g)
    alpha     = np.mean(-log_delta)
    si_rmse   = float(np.sqrt(np.mean((log_delta + alpha) ** 2)))

    A        = np.stack([p, np.ones_like(p)], axis=1)
    s, t     = np.linalg.lstsq(A, g, rcond=None)[0]
    p_affine = np.clip(s * p + t, a_min=1e-6, a_max=None)

    absrel = float(np.mean(np.abs(p_affine - g) / g))
    ratio  = np.maximum(p_affine / g, g / p_affine)
    delta1 = float(np.mean(ratio < 1.25))

    return {"si_rmse": si_rmse, "absrel": absrel, "delta1": delta1}


def compute_metrics_prescaled(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Metrics with no per-image alignment — pred is assumed already globally scaled."""
    valid = _valid_mask(pred, gt)
    if valid.sum() == 0:
        return {"si_rmse": float("nan"), "absrel": float("nan"), "delta1": float("nan")}

    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)

    log_delta = np.log(p) - np.log(g)
    alpha     = np.mean(-log_delta)
    si_rmse   = float(np.sqrt(np.mean((log_delta + alpha) ** 2)))

    absrel = float(np.mean(np.abs(p - g) / g))
    ratio  = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))

    return {"si_rmse": si_rmse, "absrel": absrel, "delta1": delta1}


def _compute_global_scale(preds: list, gts: list) -> float:
    """Median-ratio global scale: median(all_valid_gt) / median(all_valid_pred)."""
    gt_acc, pred_acc = [], []
    for p, g in zip(preds, gts):
        mask = _valid_mask(p, g)
        if mask.sum() < 10:
            continue
        gt_acc.append(g[mask][::4])
        pred_acc.append(p[mask][::4])
    if not gt_acc:
        return 1.0
    return float(np.median(np.concatenate(gt_acc)) / np.median(np.concatenate(pred_acc)))


# ── Model building ────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device):
    source = _detect_source(cfg)
    student_type = cfg.get("student_type", "resnet34")

    if source == "sf_eagle":
        # Use EAGLEDepthModel from models.py (SFT checkpoint)
        SFEDepthModel = EAGLEDepthModel
        SFEDecodeHead = DepthDecodeHead

        if student_type == "mobilevit":
            mobilevit_model_id = cfg.get("mobilevit_model_id", "apple/mobilevit-xx-small")
            encoder = MobileViTModel.from_pretrained(mobilevit_model_id)
            hidden_sizes = list(encoder.config.neck_hidden_sizes[1:-1])
            input_is_imagenet_normalized = False
        else:
            seg_model_id = cfg.get("seg_model_id", "nvidia/segformer-b0-finetuned-ade-512-512")
            try:
                encoder = SegformerModel.from_pretrained(seg_model_id)
            except Exception:
                encoder = SegformerForSemanticSegmentation.from_pretrained(seg_model_id).segformer
            hidden_sizes = list(encoder.config.hidden_sizes)
            input_is_imagenet_normalized = True

        depth_head = SFEDecodeHead(
            hidden_sizes=hidden_sizes,
            decoder_hidden_size=cfg.get("decoder_hidden_size", 256),
            dropout=cfg.get("decoder_dropout", 0.1),
        )
        model = SFEDepthModel(
            encoder=encoder,
            depth_head=depth_head,
            hidden_sizes=hidden_sizes,
            eam_stages=cfg.get("eam_stages", [3]),
            eam_k=cfg.get("eam_k", 4),
            num_clusters=cfg.get("num_clusters", 10),
            eam_sigma_color=cfg.get("eam_sigma_color", 0.2),
            multi_layer_affinity=cfg.get("multi_layer_affinity", False),
            differentiable_eigh=cfg.get("differentiable_eigh", False),
            input_is_imagenet_normalized=input_is_imagenet_normalized,
        ).to(device)

    else:
        # ResNet34 or SegFormer with optional ASPP. These are older checkpoints
        # that have `use_aspp` in their config or `student_type == "resnet34"`.
        # New sf_eagle checkpoints never hit this path.
        if student_type == "segformer":
            seg_model_id = cfg.get("seg_model_id", "nvidia/segformer-b0-finetuned-ade-512-512")
            try:
                encoder = SegformerModel.from_pretrained(seg_model_id)
            except Exception:
                encoder = SegformerForSemanticSegmentation.from_pretrained(seg_model_id).segformer
            hidden_sizes = list(encoder.config.hidden_sizes)
        else:
            encoder      = ResNet34EncoderWrapper(pretrained=False)
            hidden_sizes = _RESNET34_HIDDEN_SIZES

        depth_head = ResNet34DepthDecodeHead(
            hidden_sizes=hidden_sizes,
            decoder_hidden_size=cfg.get("decoder_hidden_size", 256),
            dropout=cfg.get("decoder_dropout", 0.1),
            use_aspp=cfg.get("use_aspp", False),
        )
        model = EAGLEDepthModel(
            encoder=encoder,
            depth_head=depth_head,
            hidden_sizes=hidden_sizes,
            eam_stages=cfg.get("eam_stages", [3]),
            eam_k=cfg.get("eam_k", 4),
            num_clusters=cfg.get("num_clusters", 10),
            eam_sigma_color=cfg.get("eam_sigma_color", 0.2),
            multi_layer_affinity=cfg.get("multi_layer_affinity", False),
            differentiable_eigh=cfg.get("differentiable_eigh", False),
            input_is_imagenet_normalized=True,
        ).to(device)

    return model


def load_weights(model, ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return ckpt


# ── Standalone .pth helpers ───────────────────────────────────────────────────

_BASELINE_VARIANTS = ("resnet18", "resnet34", "resnet50", "unet")


def _infer_baseline_variant(pth_path: Path) -> str:
    stem = pth_path.stem.lower()
    for variant in _BASELINE_VARIANTS:
        if variant in stem:
            return variant
    raise ValueError(
        f"Cannot infer model variant from '{pth_path.name}'. "
        f"Expected filename to contain one of: {_BASELINE_VARIANTS}"
    )


def load_standalone_model(pth_path: Path, device: torch.device):
    """Build and load a standalone best_*.pth model (ResNetDepth / UNet).

    The weights were saved via ``torch.save(torch.compile(model).state_dict(), ...)``,
    so every key carries an ``_orig_mod.`` prefix that must be stripped.
    """
    variant = _infer_baseline_variant(pth_path)
    model = build_baseline_model(variant).to(device)

    raw_sd = torch.load(pth_path, map_location=device, weights_only=False)
    cleaned_sd = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in raw_sd.items()
    }
    model.load_state_dict(cleaned_sd)
    return model, variant


def evaluate_standalone(pth_path: Path, data_dir: Path, device: torch.device,
                        metric_protocol: str = "affine") -> dict:
    """Evaluate a standalone best_*.pth checkpoint (no run_config.json needed)."""
    print(f"\n{'='*60}")
    print(f"Checkpoint : {pth_path}  (standalone .pth)")

    model, variant = load_standalone_model(pth_path, device)
    print(f"Variant    : {variant}")
    print(f"Input      : raw [0,1], size=256, no ImageNet norm")
    print(f"Protocol   : {metric_protocol}")
    model.eval()

    input_size = 256
    rgb_paths  = sorted(data_dir.glob("bench_*_rgb.png"))
    if not rgb_paths:
        raise RuntimeError(f"No bench_*_rgb.png files found in {data_dir}")

    pairs: list = []
    print(f"\nInference on {len(rgb_paths)} pairs ...  (imagenet_norm=False)\n")

    with torch.no_grad():
        for i, rgb_path in enumerate(rgb_paths):
            depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
            if not depth_path.exists():
                print(f"  [{i+1}/{len(rgb_paths)}] missing depth for {rgb_path.name}, skip")
                continue

            gt    = np.load(str(depth_path)).astype(np.float32)
            rgb_t = preprocess_rgb(rgb_path, input_size, normalize=False).to(device)
            pred_t = model(rgb_t)

            gh, gw = gt.shape
            if pred_t.dim() == 3:
                pred_t = pred_t.unsqueeze(1)
            pred_np = (
                F.interpolate(
                    pred_t.float(),
                    size=(gh, gw),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            pairs.append((pred_np, gt, rgb_path.name))

    scale = 1.0
    if metric_protocol == "global_median":
        scale = _compute_global_scale([p for p, _, _ in pairs], [g for _, g, _ in pairs])
        print(f"Global scale   : {scale:.6f}")

    all_metrics: list = []
    print()
    for i, (pred_np, gt, name) in enumerate(pairs):
        if metric_protocol == "global_median":
            m = compute_metrics_prescaled(pred_np * scale, gt)
        else:
            m = compute_metrics(pred_np, gt)
        all_metrics.append(m)

        depth_coverage = float(np.mean(gt > 1e-3) * 100)
        print(f"  [{i+1:3d}/{len(pairs)}] {name}  "
              f"SI-RMSE={m['si_rmse']:.4f}  AbsRel={m['absrel']:.4f}  "
              f"d1={m['delta1']:.3f}  cov={depth_coverage:.1f}%")

    def _agg(key: str) -> list:
        return [m[key] for m in all_metrics if not np.isnan(m[key])]

    si_vals  = _agg("si_rmse")
    abs_vals = _agg("absrel")
    d1_vals  = _agg("delta1")

    return {
        "label":    pth_path.stem,
        "n":        len(all_metrics),
        "si_rmse":  float(np.mean(si_vals))  if si_vals  else float("nan"),
        "absrel":   float(np.mean(abs_vals)) if abs_vals else float("nan"),
        "delta1":   float(np.mean(d1_vals))  if d1_vals  else float("nan"),
        "protocol": metric_protocol,
    }


# ── Single-checkpoint evaluation ──────────────────────────────────────────────

def evaluate_checkpoint(ckpt_dir: Path, data_dir: Path, device: torch.device,
                        metric_protocol: str = "affine") -> dict:
    """Run inference for one checkpoint dir; return aggregated metrics + label."""
    cfg_path  = ckpt_dir / "run_config.json"
    ckpt_path = ckpt_dir / "best_checkpoint.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {ckpt_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_checkpoint.pt not found in {ckpt_dir}")

    with open(cfg_path) as fh:
        cfg = json.load(fh)

    source = _detect_source(cfg)
    label  = ckpt_dir.name

    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_dir}")
    print(f"Source     : {source}")
    print(f"student_type={cfg.get('student_type')}  "
          f"eam_stages={cfg.get('eam_stages')}  "
          f"input_size={cfg.get('input_size', 256)}")
    print(f"Protocol   : {metric_protocol}")

    model = build_model(cfg, device)
    ckpt  = load_weights(model, ckpt_path, device)
    print(f"Loaded — epoch {ckpt.get('epoch', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    model.eval()

    input_size    = cfg.get("input_size", 256)
    imagenet_norm = cfg.get("student_type", "resnet34") != "mobilevit"
    rgb_paths     = sorted(data_dir.glob("bench_*_rgb.png"))
    if not rgb_paths:
        raise RuntimeError(f"No bench_*_rgb.png files found in {data_dir}")

    pairs: list = []
    print(f"\nInference on {len(rgb_paths)} pairs ...  "
          f"(imagenet_norm={imagenet_norm})\n")

    with torch.no_grad():
        for i, rgb_path in enumerate(rgb_paths):
            depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
            if not depth_path.exists():
                print(f"  [{i+1}/{len(rgb_paths)}] missing depth for {rgb_path.name}, skip")
                continue

            gt = np.load(str(depth_path)).astype(np.float32)

            rgb_t = preprocess_rgb(rgb_path, input_size, normalize=imagenet_norm).to(device)
            out   = model(rgb_t)
            pred  = out["depth"]

            gh, gw = gt.shape
            pred_np = (
                F.interpolate(
                    pred.unsqueeze(1).float(),
                    size=(gh, gw),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            pairs.append((pred_np, gt, rgb_path.name))

    scale = 1.0
    if metric_protocol == "global_median":
        scale = _compute_global_scale([p for p, _, _ in pairs], [g for _, g, _ in pairs])
        print(f"Global scale   : {scale:.6f}")

    all_metrics: list = []
    print()
    for i, (pred_np, gt, name) in enumerate(pairs):
        if metric_protocol == "global_median":
            m = compute_metrics_prescaled(pred_np * scale, gt)
        else:
            m = compute_metrics(pred_np, gt)
        all_metrics.append(m)

        depth_coverage = float(np.mean(gt > 1e-3) * 100)
        print(f"  [{i+1:3d}/{len(pairs)}] {name}  "
              f"SI-RMSE={m['si_rmse']:.4f}  AbsRel={m['absrel']:.4f}  "
              f"d1={m['delta1']:.3f}  cov={depth_coverage:.1f}%")

    def _agg(key: str) -> list:
        return [m[key] for m in all_metrics if not np.isnan(m[key])]

    si_vals  = _agg("si_rmse")
    abs_vals = _agg("absrel")
    d1_vals  = _agg("delta1")

    return {
        "label":    label,
        "n":        len(all_metrics),
        "si_rmse":  float(np.mean(si_vals))  if si_vals  else float("nan"),
        "absrel":   float(np.mean(abs_vals)) if abs_vals else float("nan"),
        "delta1":   float(np.mean(d1_vals))  if d1_vals  else float("nan"),
        "protocol": metric_protocol,
    }


# ── MobileNet / TinyDepthUNet checkpoint evaluation ───────────────────────────

def _is_mobilenet_ckpt(ckpt_dir: Path) -> bool:
    has_ckpt = (ckpt_dir / "checkpoint_best.pth").exists() or (ckpt_dir / "checkpoint.pth").exists()
    return has_ckpt and (ckpt_dir / "config.yaml").exists()


def evaluate_mobilenet_checkpoint(ckpt_dir: Path, data_dir: Path, device: torch.device,
                                   metric_protocol: str = "affine") -> dict:
    """Evaluate a TinyDepthUNet (MobileNetV3-Small) checkpoint."""
    cfg_path  = ckpt_dir / "config.yaml"
    ckpt_path = (ckpt_dir / "checkpoint_best.pth") if (ckpt_dir / "checkpoint_best.pth").exists() \
                else (ckpt_dir / "checkpoint.pth")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {ckpt_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Neither checkpoint_best.pth nor checkpoint.pth found in {ckpt_dir}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    label = ckpt_dir.name

    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_dir}")
    print(f"Source     : TinyDepthUNet (MobileNetV3-Small)")
    print(f"bottleneck={mcfg.get('bottleneck_channels', 64)}  "
          f"decoder={mcfg.get('decoder_channels')}  "
          f"input_size={dcfg.get('image_size', 256)}")
    print(f"Protocol   : {metric_protocol}")

    model = TinyDepthUNet(
        bottleneck_channels=mcfg.get("bottleneck_channels", 64),
        decoder_channels=tuple(mcfg.get("decoder_channels", [64, 48, 32, 16])),
        eam_scales=tuple(mcfg.get("eam_scales", [])),
        eam_k=mcfg.get("eam_k", 4),
        num_clusters=mcfg.get("num_clusters", 10),
        eam_sigma_color=mcfg.get("eam_sigma_color", 0.2),
        use_residual_eam=mcfg.get("use_residual_eam", False),
        pretrained_encoder=False,
        use_aspp=mcfg.get("use_aspp", True),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded — epoch {ckpt.get('epoch', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    model.eval()

    input_size = dcfg.get("image_size", 256)
    rgb_paths  = sorted(data_dir.glob("bench_*_rgb.png"))
    if not rgb_paths:
        raise RuntimeError(f"No bench_*_rgb.png files found in {data_dir}")

    pairs: list = []
    print(f"\nInference on {len(rgb_paths)} pairs ...  (imagenet_norm=True, log_depth->exp)\n")

    with torch.no_grad():
        for i, rgb_path in enumerate(rgb_paths):
            depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
            if not depth_path.exists():
                print(f"  [{i+1}/{len(rgb_paths)}] missing depth for {rgb_path.name}, skip")
                continue

            gt    = np.load(str(depth_path)).astype(np.float32)
            rgb_t = preprocess_rgb(rgb_path, input_size, normalize=True).to(device)
            out   = model(rgb_t)
            pred  = torch.exp(out["log_depth"])

            gh, gw = gt.shape
            pred_np = (
                F.interpolate(
                    pred.float(),
                    size=(gh, gw),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            pairs.append((pred_np, gt, rgb_path.name))

    scale = 1.0
    if metric_protocol == "global_median":
        scale = _compute_global_scale([p for p, _, _ in pairs], [g for _, g, _ in pairs])
        print(f"Global scale   : {scale:.6f}")

    all_metrics: list = []
    print()
    for i, (pred_np, gt, name) in enumerate(pairs):
        if metric_protocol == "global_median":
            m = compute_metrics_prescaled(pred_np * scale, gt)
        else:
            m = compute_metrics(pred_np, gt)
        all_metrics.append(m)

        depth_coverage = float(np.mean(gt > 1e-3) * 100)
        print(f"  [{i+1:3d}/{len(pairs)}] {name}  "
              f"SI-RMSE={m['si_rmse']:.4f}  AbsRel={m['absrel']:.4f}  "
              f"d1={m['delta1']:.3f}  cov={depth_coverage:.1f}%")

    def _agg(key: str) -> list:
        return [m[key] for m in all_metrics if not np.isnan(m[key])]

    si_vals  = _agg("si_rmse")
    abs_vals = _agg("absrel")
    d1_vals  = _agg("delta1")

    return {
        "label":    label,
        "n":        len(all_metrics),
        "si_rmse":  float(np.mean(si_vals))  if si_vals  else float("nan"),
        "absrel":   float(np.mean(abs_vals)) if abs_vals else float("nan"),
        "delta1":   float(np.mean(d1_vals))  if d1_vals  else float("nan"),
        "protocol": metric_protocol,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH3D benchmark inference + evaluation")
    p.add_argument(
        "--checkpoint_dirs",
        nargs="+",
        default=[_DEFAULT_CKPT_DIR],
        help=(
            "One or more checkpoint directories (containing run_config.json + "
            "best_checkpoint.pt) OR standalone best_*.pth files."
        ),
    )
    p.add_argument("--data_dir", type=str, default=_DEFAULT_DATA_DIR)
    p.add_argument("--device",   type=str, default="")
    p.add_argument(
        "--metric_protocol",
        choices=["affine", "global_median"],
        default="affine",
        help=(
            "affine: per-image least-squares scale+shift alignment, matching DAV2 eval (default). "
            "global_median: single global median ratio across all images."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    device   = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    results = []
    for ckpt_path_str in args.checkpoint_dirs:
        ckpt_path = Path(ckpt_path_str)
        if ckpt_path.is_file():
            results.append(evaluate_standalone(ckpt_path, data_dir, device,
                                               metric_protocol=args.metric_protocol))
        elif _is_mobilenet_ckpt(ckpt_path):
            results.append(evaluate_mobilenet_checkpoint(ckpt_path, data_dir, device,
                                                          metric_protocol=args.metric_protocol))
        else:
            results.append(evaluate_checkpoint(ckpt_path, data_dir, device,
                                               metric_protocol=args.metric_protocol))

    proto = args.metric_protocol
    print(f"\n\n{'='*70}")
    print(f"SUMMARY  (metric_protocol={proto})")
    if proto == "affine":
        print(f"Note: AbsRel and delta1 use affine (scale+shift) alignment, matching DAV2 eval.")
        print(f"      delta1 is a fraction in [0,1] — multiply by 100 for %. DAV2 ETH3D: 0.851–0.885")
    else:
        print(f"Note: AbsRel and delta1 use a single global median scale (classic NYUv2 protocol).")
        print(f"      delta1 is a fraction in [0,1] — multiply by 100 for %.")
    print(f"{'─'*70}")
    hdr = f"{'Model':<40}  {'n':>4}  {'SI-RMSE':>8}  {'AbsRel':>8}  {'d1':>8}"
    print(hdr)
    print(f"{'─'*70}")
    for r in results:
        print(
            f"{r['label']:<40}  {r['n']:>4}  "
            f"{r['si_rmse']:>8.4f}  {r['absrel']:>8.4f}  {r['delta1']:>8.4f}"
        )
    print(f"{'─'*70}")
    print(f"[DAV2 ETH3D ref]  AbsRel: 0.131–0.142   delta1: 0.851–0.885")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
