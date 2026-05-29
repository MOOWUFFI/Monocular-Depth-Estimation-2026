#!/usr/bin/env python3
"""Training entrypoint for the MobileViT-XX-Small + EAGLE approach.

Total loss = silog_weight * SILog
           + eig_cluster_weight * L_eig
           + within_cluster_weight * L_within_cluster

What is trained:
  Always:    DepthDecodeHead + EigenAggregationModules (EAMs)
  Optional:  MobileViT encoder layers (at a reduced LR via --unfreeze_encoder)

MobileViT input is raw [0, 1] (no ImageNet normalization).

Usage (from the repository root):
    # From scratch (encoder loaded from the pretrained MobileViT-XX-Small)
    python -m approaches.Mobilevit.train.train --from_scratch \
        --eam_stages 3 4 --output_dir approaches/Mobilevit/results/eagle_eam3s4

    # From a Stage-2 distilled encoder + head
    python -m approaches.Mobilevit.train.train \
        --stage2_encoder <dir>/best_encoder \
        --stage2_head    <dir>/best_depth_head.pt \
        --eam_stages 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast, GradScaler
    _AMP_NEW_API = True
except ImportError:  # pragma: no cover - legacy torch fallback
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW_API = False

from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import MobileViTConfig, MobileViTModel

from scripts.constants import TRAIN_DIR
from approaches.Mobilevit.model import DepthDecodeHead, EAGLEDepthModel
from approaches.Mobilevit.train.dataset import DepthDataset, build_data_splits
from approaches.Mobilevit.train.losses import EagleDepthLoss
from approaches.Mobilevit.train.schedulers import build_scheduler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Training epoch ──────────────────────────────────────────────────────────

def run_eagle_epoch(
    model:          EAGLEDepthModel,
    criterion:      EagleDepthLoss,
    loader:         DataLoader,
    optimizer:      Optional[AdamW],
    scheduler:      Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler:         GradScaler,
    device:         torch.device,
    use_amp:        bool,
    grad_accum:     int,
    is_train:       bool,
    max_grad_norm:  float = 1.0,
    log_interval:   int   = 50,
) -> Tuple[float, Dict[str, float], float]:
    """One epoch of EAGLE fine-tuning.

    Returns (mean_total_loss, mean_component_dict, elapsed_seconds).
    """
    model.depth_head.train(is_train)
    for eam in model.eams.values():
        eam.train(is_train)
    if hasattr(model, "affinity_projs"):
        for proj in model.affinity_projs.values():
            proj.train(is_train)
    encoder_trainable = any(p.requires_grad for p in model.encoder.parameters())
    model.encoder.train(is_train and encoder_trainable)

    running_total = 0.0
    running_components: Dict[str, float] = {}
    n_steps    = 0
    accum_step = 0
    t0         = time.time()

    amp_ctx = (
        (lambda: autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp))
        if _AMP_NEW_API
        else (lambda: autocast(enabled=use_amp))
    )
    grad_ctx = torch.enable_grad if is_train else torch.no_grad

    with grad_ctx():
        if is_train:
            optimizer.zero_grad()

        for step, (rgb, gt_depth) in enumerate(loader):
            rgb      = rgb.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)

            with amp_ctx():
                out    = model(rgb)
                pred   = out["depth"]    # (B, H, W)
                logits = out["logits"]   # dict[int, (B, N, K)]

                if pred.shape[-2:] != gt_depth.shape[-2:]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=gt_depth.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                loss, comps = criterion(pred.float(), gt_depth.float(), logits)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                log.warning(
                    "Non-finite loss (%.4g) at step %d — skipping", loss_val, step
                )
                if is_train and accum_step > 0:
                    scaler.update()
                    optimizer.zero_grad()
                    accum_step = 0
                continue

            if is_train:
                scaler.scale(loss / grad_accum).backward()
                accum_step += 1

                if accum_step % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    trainable_params = [
                        p for p in model.parameters() if p.requires_grad
                    ]
                    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    accum_step = 0
                    if scaler.get_scale() == scale_before:
                        scheduler.step()

            running_total += loss_val
            for k, v in comps.items():
                running_components[k] = running_components.get(k, 0.0) + v.item()
            n_steps += 1

            if (step + 1) % log_interval == 0:
                elapsed  = time.time() - t0
                done     = step + 1
                total    = len(loader)
                sps      = done / max(elapsed, 1e-9)
                eta_secs = (total - done) / max(sps, 1e-9)
                phase    = "TRAIN" if is_train else "VAL  "
                lr_str   = (
                    f"lr={optimizer.param_groups[-1]['lr']:.2e}  "
                    if is_train and optimizer else ""
                )
                avg = running_total / n_steps
                comp_str = "  ".join(
                    f"{k}={v / n_steps:.4f}"
                    for k, v in running_components.items()
                )
                log.info(
                    "[EAGLE-%s] step %4d/%4d (%5.1f%%) | total=%.5f | %s | %s"
                    "elapsed %s | ETA %s",
                    phase, done, total,
                    100.0 * done / total,
                    avg, comp_str, lr_str,
                    _fmt_duration(elapsed),
                    _fmt_duration(eta_secs),
                )

    denom = max(n_steps, 1)
    return (
        running_total / denom,
        {k: v / denom for k, v in running_components.items()},
        time.time() - t0,
    )


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: EAGLEDepthModel,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                 epoch,
            "best_val_loss":         best_val_loss,
            "encoder_state_dict":    model.encoder.state_dict(),
            "depth_head_state_dict": model.depth_head.state_dict(),
            "eam_state_dict":        {k: v.state_dict() for k, v in model.eams.items()},
            "affinity_proj_state_dict": (
                {k: v.state_dict() for k, v in model.affinity_projs.items()}
                if hasattr(model, "affinity_projs") else {}
            ),
            "optimizer_state_dict":  optimizer.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "scaler_state_dict":     scaler.state_dict(),
            "args":                  vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: EAGLEDepthModel,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    """Returns (epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.depth_head.load_state_dict(ckpt["depth_head_state_dict"])
    for key, eam in model.eams.items():
        if key in ckpt["eam_state_dict"]:
            eam.load_state_dict(ckpt["eam_state_dict"][key])
    if hasattr(model, "affinity_projs") and ckpt.get("affinity_proj_state_dict"):
        for key, proj in model.affinity_projs.items():
            if key in ckpt["affinity_proj_state_dict"]:
                proj.load_state_dict(ckpt["affinity_proj_state_dict"][key])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EAGLE-augmented fine-tuning of the MobileViT-XX-Small depth model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage-2 checkpoint / scratch mode ───────────────────────────────────
    p.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
        help=(
            "Train EAGLE directly from the pretrained MobileViT checkpoint "
            "(no Stage-2 distilled encoder/head). When set, --stage2_encoder "
            "and --stage2_head are ignored and the encoder is loaded from "
            "--mobilevit_model_id while the depth head is randomly initialised."
        ),
    )
    p.add_argument(
        "--stage2_encoder",
        type=str,
        default="",
        help=(
            "Path to a Stage-2 best_encoder directory (a HuggingFace "
            "MobileViTModel save). Required unless --from_scratch is set."
        ),
    )
    p.add_argument(
        "--stage2_head",
        type=str,
        default="",
        help=(
            "Path to a Stage-2 best_depth_head.pt file (state-dict of "
            "DepthDecodeHead). Required unless --from_scratch is set."
        ),
    )

    # ── Model config ─────────────────────────────────────────────────────────
    p.add_argument(
        "--mobilevit_model_id",
        type=str,
        default="apple/mobilevit-xx-small",
        help="HuggingFace model ID / local path for the MobileViT config + weights.",
    )

    # ── EAM configuration ────────────────────────────────────────────────────
    p.add_argument(
        "--eam_stages",
        type=int,
        nargs="+",
        default=[3, 4],
        help=(
            "Encoder stage indices at which to attach EAMs. "
            "MobileViT-XXS has stages 0-4: stages 3 (16x16) and 4 (8x8) "
            "are recommended (small patch grids)."
        ),
    )
    p.add_argument("--eam_k", type=int, default=4,
                   help="Number of non-trivial eigenvectors (k) per EAM.")
    p.add_argument("--num_clusters", type=int, default=10,
                   help="Number of learnable cluster centers in each EAM.")
    p.add_argument("--eam_sigma_color", type=float, default=0.2,
                   help="sigma for the Gaussian color affinity kernel.")
    p.add_argument(
        "--multi_layer_affinity",
        action="store_true",
        default=False,
        help="Build the semantic affinity from concatenated features of the "
             "three preceding encoder stages (adds 1x1 projection layers).",
    )
    p.add_argument(
        "--differentiable_eigh",
        action="store_true",
        default=False,
        help="Allow gradients through the eigendecomposition (experimental).",
    )

    # ── Encoder training ──────────────────────────────────────────────────────
    p.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        default=False,
        help="Unfreeze the MobileViT encoder and train it at encoder_lr_mult x base LR.",
    )
    p.add_argument("--encoder_lr_mult", type=float, default=0.1,
                   help="LR multiplier for the encoder (only used with --unfreeze_encoder).")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--input_size", type=int,  default=256,
                   help="Square input resolution.")

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--num_epochs",    type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--grad_accum",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── Loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--silog_weight",          type=float, default=1.0)
    p.add_argument("--eig_cluster_weight",     type=float, default=0.05)
    p.add_argument("--within_cluster_weight",  type=float, default=0.1)

    # ── Decoder config (must match Stage-2 checkpoint) ────────────────────────
    p.add_argument("--decoder_hidden_size", type=int,   default=256)
    p.add_argument("--decoder_dropout",     type=float, default=0.1)

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="approaches/Mobilevit/results")
    p.add_argument("--resume_from", type=str, default="",
                   help="Path to a checkpoint .pt file to resume from.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(
        "Device: %s | AMP: %s | student_type: mobilevit | EAM stages: %s | unfreeze_encoder: %s",
        device, use_amp, args.eam_stages, args.unfreeze_encoder,
    )

    # ── Output directory ──────────────────────────────────────────────────────
    stage_tag = "s".join(str(s) for s in sorted(args.eam_stages))
    if args.from_scratch:
        run_tag = (
            f"mobilevit_eagle_scratch_eam{stage_tag}_k{args.eam_k}_c{args.num_clusters}"
            f"_lr{args.lr:.0e}"
        )
    else:
        run_tag = (
            f"eagle_stage3_mobilevit_eam{stage_tag}_k{args.eam_k}_c{args.num_clusters}"
            f"_lr{args.lr:.0e}"
        )
    out_dir = Path(args.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_pairs, val_pairs = build_data_splits(args.train_dir, args.val_split, args.seed)
    log.info("Split: %d train / %d val", len(train_pairs), len(val_pairs))
    imagenet_norm = False  # MobileViT expects raw [0, 1] input
    train_ds = DepthDataset(train_pairs, args.input_size, is_train=True,
                            normalize_imagenet=imagenet_norm)
    val_ds   = DepthDataset(val_pairs,   args.input_size, is_train=False,
                            normalize_imagenet=imagenet_norm)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Determine hidden_sizes + load encoder ─────────────────────────────────
    if args.from_scratch:
        log.info("from_scratch=True — loading MobileViT encoder from %s …",
                 args.mobilevit_model_id)
        encoder = MobileViTModel.from_pretrained(args.mobilevit_model_id)
    else:
        if not args.stage2_encoder:
            raise ValueError("--stage2_encoder is required when --from_scratch is not set.")
        log.info("Loading Stage-2 MobileViT encoder from %s …", args.stage2_encoder)
        encoder_path = Path(args.stage2_encoder).resolve()
        if not encoder_path.is_dir():
            raise FileNotFoundError(f"Stage-2 encoder directory not found: {encoder_path}")
        # Some versions of transformers/huggingface_hub run a repo-ID validation
        # before checking whether the path is a local directory, causing a
        # spurious HFValidationError for any local path with more than one '/'
        # component. Work around it by loading the config and weights directly.
        mvit_config = MobileViTConfig.from_pretrained(str(encoder_path))
        encoder = MobileViTModel(mvit_config)
        sf_path  = encoder_path / "model.safetensors"
        bin_path = encoder_path / "pytorch_model.bin"
        if sf_path.exists():
            try:
                from safetensors.torch import load_file as _sf_load
                _sd = _sf_load(str(sf_path))
            except ImportError:
                _sd = torch.load(str(sf_path), map_location="cpu", weights_only=True)
        elif bin_path.exists():
            _sd = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No weights file (model.safetensors / pytorch_model.bin) found in {encoder_path}"
            )
        missing, unexpected = encoder.load_state_dict(_sd, strict=False)
        if missing:
            log.warning("Missing keys when loading MobileViT encoder: %s", missing)
        if unexpected:
            log.warning("Unexpected keys when loading MobileViT encoder: %s", unexpected)

    # neck_hidden_sizes[1:-1] are the per-encoder-layer output channels.
    hidden_sizes: List[int] = list(encoder.config.neck_hidden_sizes[1:-1])
    log.info("MobileViT hidden_sizes: %s", hidden_sizes)

    # Validate that all requested EAM stages exist in this encoder
    max_stage = len(hidden_sizes) - 1
    for s in args.eam_stages:
        if s > max_stage:
            raise ValueError(
                f"--eam_stages contains {s} but MobileViT only has stages "
                f"0-{max_stage} (hidden_sizes={hidden_sizes})"
            )

    encoder = encoder.to(device)
    log.info("Encoder loaded. Params: %d", sum(p.numel() for p in encoder.parameters()))

    # Freeze / unfreeze encoder
    for p in encoder.parameters():
        p.requires_grad_(args.unfreeze_encoder)
    if args.unfreeze_encoder:
        log.info("Encoder UNFROZEN (encoder_lr_mult=%.2f x base lr=%.1e).",
                 args.encoder_lr_mult, args.lr)
    else:
        log.info("Encoder FROZEN.")

    # ── Load / initialise depth head ──────────────────────────────────────────
    depth_head = DepthDecodeHead(
        hidden_sizes=hidden_sizes,
        decoder_hidden_size=args.decoder_hidden_size,
        dropout=args.decoder_dropout,
    )
    if args.from_scratch:
        log.info("from_scratch=True — depth head initialised with random weights.")
    else:
        if not args.stage2_head:
            raise ValueError("--stage2_head is required when --from_scratch is not set.")
        log.info("Loading Stage-2 depth head from %s …", args.stage2_head)
        head_ckpt = torch.load(args.stage2_head, map_location="cpu", weights_only=True)
        depth_head.load_state_dict(head_ckpt)
    depth_head = depth_head.to(device)

    # ── Build EAGLE model ─────────────────────────────────────────────────────
    model = EAGLEDepthModel(
        encoder=encoder,
        depth_head=depth_head,
        hidden_sizes=hidden_sizes,
        eam_stages=args.eam_stages,
        eam_k=args.eam_k,
        num_clusters=args.num_clusters,
        eam_sigma_color=args.eam_sigma_color,
        multi_layer_affinity=args.multi_layer_affinity,
        differentiable_eigh=args.differentiable_eigh,
        input_is_imagenet_normalized=imagenet_norm,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    log.info("Model: %d total params | %d trainable", n_total, n_trainable)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = EagleDepthLoss(
        silog_weight=args.silog_weight,
        eig_cluster_weight=args.eig_cluster_weight,
        within_cluster_weight=args.within_cluster_weight,
    ).to(device)
    log.info("Loss weights — silog=%.2f  eig_cluster=%.3f  within_cluster=%.3f",
             args.silog_weight, args.eig_cluster_weight, args.within_cluster_weight)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    enc_lr_mult = args.encoder_lr_mult if args.unfreeze_encoder else 0.0
    optimizer = AdamW(
        model.param_groups(args.lr, enc_lr_mult, args.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps     = steps_per_epoch * args.num_epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler       = build_scheduler(optimizer, total_steps, warmup_steps)
    scaler          = GradScaler(enabled=use_amp)
    log.info("AdamW base_lr=%.1e wd=%.1e total_steps=%d warmup=%d eff_batch=%d",
             args.lr, args.weight_decay, total_steps, warmup_steps,
             args.batch_size * args.grad_accum)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    history: List[dict] = []

    if args.resume_from and Path(args.resume_from).exists():
        log.info("Resuming from %s …", args.resume_from)
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume_from), model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        log.info("Resumed: starting at epoch %d, best_val_loss=%.5f",
                 start_epoch, best_val_loss)
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as fh:
                history = json.load(fh)

    # ── Save run config ───────────────────────────────────────────────────────
    with open(out_dir / "run_config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_ckpt_path   = out_dir / "best_checkpoint.pt"
    latest_ckpt_path = out_dir / "latest_checkpoint.pt"
    best_encoder_dir = out_dir / "best_encoder"
    best_head_path   = out_dir / "best_depth_head.pt"
    best_eam_path    = out_dir / "best_eams.pt"

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss, train_comps, t_train = run_eagle_epoch(
            model=model, criterion=criterion, loader=train_loader,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            device=device, use_amp=use_amp, grad_accum=args.grad_accum,
            is_train=True, max_grad_norm=args.max_grad_norm,
        )

        val_loss, val_comps, t_val = run_eagle_epoch(
            model=model, criterion=criterion, loader=val_loader,
            optimizer=None, scheduler=None, scaler=scaler,
            device=device, use_amp=use_amp, grad_accum=args.grad_accum,
            is_train=False,
        )

        train_comp_str = "  ".join(f"{k}={v:.4f}" for k, v in train_comps.items())
        val_comp_str   = "  ".join(f"{k}={v:.4f}" for k, v in val_comps.items())
        log.info(
            "[Epoch %3d/%d] train_total=%.5f (%s) | val_total=%.5f (%s) | "
            "train_t=%s  val_t=%s",
            epoch, args.num_epochs,
            train_loss, train_comp_str,
            val_loss,   val_comp_str,
            _fmt_duration(t_train), _fmt_duration(t_val),
        )

        history.append({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "train_comps": train_comps,
            "val_comps":   val_comps,
            "lr":          optimizer.param_groups[-1]["lr"],
        })
        with open(out_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        save_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler,
                        epoch, best_val_loss, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler,
                            epoch, best_val_loss, args)
            best_encoder_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(str(best_encoder_dir))
            torch.save(model.depth_head.state_dict(), best_head_path)
            torch.save(
                {
                    "eams": {k: eam.state_dict() for k, eam in model.eams.items()},
                    "affinity_projs": (
                        {k: v.state_dict() for k, v in model.affinity_projs.items()}
                        if hasattr(model, "affinity_projs") else {}
                    ),
                },
                best_eam_path,
            )
            log.info("  New best val loss: %.5f  (epoch %d)  -> %s",
                     best_val_loss, epoch, out_dir)

    log.info("EAGLE fine-tuning complete. Best val total loss: %.5f", best_val_loss)


if __name__ == "__main__":
    main()
