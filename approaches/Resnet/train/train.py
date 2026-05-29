#!/usr/bin/env python3
"""Training entrypoint for the ResNet34 + ASPP + EAGLE depth model.

One script, five experiments (see experiments/*.sh):

    scratch              : silog | random-init encoder       | no ASPP/EAM/VN
    base                 : silog | ImageNet-pretrained        | no ASPP/EAM/VN
    base_aspp            : silog | pretrained + ASPP
    base_aspp_eagle      : silog + eig + within | + EAM stages
    base_aspp_eagle_vn   : + virtual-normal loss

Layers are additive and gated by flags:
    --pretrained_encoder / --no_pretrained_encoder   ImageNet init (resnet34)
    --use_aspp                                        ASPP in the decode head
    --use_eam --eam_stages ...                        EAGLE EAMs + cluster losses
    --vnl_weight                                      virtual-normal loss weight

Usage (from the repository root):
    python -m approaches.Resnet.train.train --from_scratch \\
        --use_aspp --use_eam --eam_stages 2 3 --unfreeze_encoder \\
        --batch_size 16 --lr 1e-4 --num_epochs 10 --output_dir checkpoints/full
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
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW_API = False

from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.constants import TRAIN_DIR
from approaches.Resnet.train.dataset import DepthDataset, build_data_splits
from approaches.Resnet.train.losses import EagleDepthLoss
from approaches.Resnet.model import (
    DepthDecodeHead,
    EAGLEDepthModel,
    ResNet34EncoderWrapper,
    _RESNET34_HIDDEN_SIZES,
)
from approaches.Resnet.train.schedulers import build_scheduler

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


# ── Training epoch ────────────────────────────────────────────────────────────

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
    n_steps   = 0
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
                pred   = out["depth"]
                logits = out["logits"]

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
                log.warning("Non-finite loss (%.4g) at step %d — skipping", loss_val, step)
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
            "epoch":               epoch,
            "best_val_loss":       best_val_loss,
            "encoder_state_dict":  model.encoder.state_dict(),
            "depth_head_state_dict": model.depth_head.state_dict(),
            "eam_state_dict":      {k: v.state_dict() for k, v in model.eams.items()},
            "affinity_proj_state_dict": (
                {k: v.state_dict() for k, v in model.affinity_projs.items()}
                if hasattr(model, "affinity_projs") else {}
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "args":                 vars(args),
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
        description="EAGLE-augmented fine-tuning of ResNet34.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage-2 checkpoint / scratch mode ───────────────────────────────────
    p.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
        help="Train directly from (optionally pretrained) weights without a Stage-2 checkpoint.",
    )
    p.add_argument(
        "--pretrained_encoder",
        dest="pretrained_encoder",
        action="store_true",
        default=True,
        help="Initialize the resnet34 encoder from ImageNet weights (default).",
    )
    p.add_argument(
        "--no_pretrained_encoder",
        dest="pretrained_encoder",
        action="store_false",
        help="Random-init the resnet34 encoder (true from-scratch run).",
    )
    p.add_argument(
        "--stage2_encoder",
        type=str,
        default="",
        help="Path to the Stage-2 best_encoder directory. Required unless --from_scratch.",
    )
    p.add_argument(
        "--stage2_head",
        type=str,
        default="",
        help="Path to the Stage-2 best_depth_head.pt file. Required unless --from_scratch.",
    )

    # ── EAM configuration ─────────────────────────────────────────────────────
    p.add_argument(
        "--use_eam",
        action="store_true",
        default=False,
        help="Enable EAGLE EigenAggregationModules + cluster losses. If unset, "
             "no EAMs are built and the cluster loss weights are forced to 0.",
    )
    p.add_argument(
        "--eam_stages",
        type=int,
        nargs="+",
        default=[3],
        help="Encoder stage indices at which to attach EAMs. Recommended: 2 and 3.",
    )
    p.add_argument("--eam_k", type=int, default=4)
    p.add_argument("--num_clusters", type=int, default=10)
    p.add_argument("--eam_sigma_color", type=float, default=0.2)
    p.add_argument("--multi_layer_affinity", action="store_true", default=False)
    p.add_argument("--differentiable_eigh", action="store_true", default=False)

    # ── Encoder training ──────────────────────────────────────────────────────
    p.add_argument("--unfreeze_encoder", action="store_true", default=False)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--train_dir", type=str, default=TRAIN_DIR,
                   help="RGB source dir with <basename>_rgb.png.")
    p.add_argument("--gt_depth_dir", type=str, default=None,
                   help="Override GT depth source (used for cleaned-GT runs). "
                        "When unset, depth loads from --train_dir.")
    p.add_argument("--val_split",  type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--input_size", type=int, default=256)

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--num_epochs",      type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)

    # ── Loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--silog_weight", type=float, default=1.0)
    p.add_argument("--eig_cluster_weight", type=float, default=0.05)
    p.add_argument("--within_cluster_weight", type=float, default=0.1)
    p.add_argument("--vnl_weight", type=float, default=5.0,
                   help="Virtual-normal loss weight. Set 0.0 to disable VN.")

    # ── Decoder config ────────────────────────────────────────────────────────
    p.add_argument("--decoder_hidden_size", type=int, default=256)
    p.add_argument("--decoder_dropout", type=float, default=0.1)
    p.add_argument("--use_aspp", action="store_true", default=False, help="Use ASPP module on deepest feature map")

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--output_dir",   type=str, default="checkpoints")
    p.add_argument("--resume_from",  type=str, default="")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # EAGLE is an opt-in layer: without --use_eam, build no EAMs and zero out the
    # cluster loss weights so the model trains on depth losses alone.
    if not args.use_eam:
        args.eam_stages = []
        args.eig_cluster_weight = 0.0
        args.within_cluster_weight = 0.0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(
        "Device: %s | AMP: %s | ASPP: %s | EAM stages: %s | unfreeze_encoder: %s",
        device, use_amp, args.use_aspp, args.eam_stages, args.unfreeze_encoder,
    )

    # ── Output directory ──────────────────────────────────────────────────────
    stage_tag = "s".join(str(s) for s in sorted(args.eam_stages))
    mode_prefix = "resnet34_eagle_scratch" if args.from_scratch else "eagle_stage3_resnet34"
    aspp_tag = "_aspp" if args.use_aspp else ""
    run_tag = f"{mode_prefix}{aspp_tag}_eam{stage_tag}_k{args.eam_k}_c{args.num_clusters}_lr{args.lr:.0e}"

    out_dir = Path(args.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── Data ──────────────────────────────────────────────────────────────────
    depth_dir = args.gt_depth_dir if args.gt_depth_dir else args.train_dir
    train_pairs, val_pairs = build_data_splits(args.train_dir, depth_dir, args.val_split, args.seed)

    # ResNet34 expects ImageNet normalisation.
    imagenet_norm = True

    train_ds = DepthDataset(train_pairs, args.input_size, is_train=True, normalize_imagenet=imagenet_norm)
    val_ds   = DepthDataset(val_pairs,   args.input_size, is_train=False, normalize_imagenet=imagenet_norm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Determine hidden_sizes ────────────────────────────────────────────────
    hidden_sizes = _RESNET34_HIDDEN_SIZES
    log.info("Resnet34 hidden_sizes: %s", hidden_sizes)

    max_stage = len(hidden_sizes) - 1
    for s in args.eam_stages:
        if s > max_stage:
            raise ValueError(
                f"--eam_stages contains {s} but resnet34 only has "
                f"stages 0-{max_stage} (hidden_sizes={hidden_sizes})"
            )

    # ── Load encoder ──────────────────────────────────────────────────────────
    if args.from_scratch:
        log.info("from_scratch=True — loading resnet34 encoder (pretrained=%s)...", args.pretrained_encoder)
        encoder = ResNet34EncoderWrapper(pretrained=args.pretrained_encoder)
    else:
        if not args.stage2_encoder:
            raise ValueError("--stage2_encoder is required when --from_scratch is not set.")
        log.info("Loading Stage-2 resnet34 encoder from %s …", args.stage2_encoder)
        encoder_path = Path(args.stage2_encoder).resolve()
        encoder = ResNet34EncoderWrapper.from_pretrained(str(encoder_path), pretrained=False)

    encoder = encoder.to(device)
    log.info("Encoder loaded. Params: %d", sum(p.numel() for p in encoder.parameters()))

    for p in encoder.parameters():
        p.requires_grad_(args.unfreeze_encoder)
    if args.unfreeze_encoder:
        log.info("Encoder UNFROZEN (encoder_lr_mult=%.2f × base lr=%.1e).", args.encoder_lr_mult, args.lr)
    else:
        log.info("Encoder FROZEN.")

    # ── Load / initialise depth head ──────────────────────────────────────────
    depth_head = DepthDecodeHead(
        hidden_sizes=hidden_sizes,
        decoder_hidden_size=args.decoder_hidden_size,
        dropout=args.decoder_dropout,
        use_aspp=args.use_aspp,
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
    log.info("Depth head loaded. Trainable params: %d", sum(p.numel() for p in depth_head.parameters()))

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
    eam_params  = sum(p.numel() for eam in model.eams.values() for p in eam.parameters())
    log.info("Model: %d total params | %d trainable | %d EAM params", n_total, n_trainable, eam_params)

    for s in sorted(args.eam_stages):
        h = hidden_sizes[s]
        hw = 64 >> s
        N  = hw * hw
        log.info("  EAM stage %d: %d ch, %d×%d patches (N=%d), affinity=%d×%d", s, h, hw, hw, N, N, N)

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    criterion = EagleDepthLoss(
        silog_weight=args.silog_weight,
        eig_cluster_weight=args.eig_cluster_weight,
        within_cluster_weight=args.within_cluster_weight,
        vnl_weight=args.vnl_weight,
    ).to(device)

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

    # ── Resume / Run Config ───────────────────────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float("inf")
    history: List[dict] = []

    if args.resume_from and Path(args.resume_from).exists():
        log.info("Resuming from %s …", args.resume_from)
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume_from), model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as fh:
                history = json.load(fh)

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
            "[Epoch %3d/%d] train_total=%.5f (%s) | val_total=%.5f (%s) | train_t=%s  val_t=%s",
            epoch, args.num_epochs, train_loss, train_comp_str,
            val_loss, val_comp_str, _fmt_duration(t_train), _fmt_duration(t_val),
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_comps": train_comps, "val_comps": val_comps, "lr": optimizer.param_groups[-1]["lr"],
        })
        with open(out_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        save_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_loss, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_val_loss, args)

            best_encoder_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model.encoder, "save_pretrained"):
                model.encoder.save_pretrained(str(best_encoder_dir))
            else:
                torch.save(model.encoder.state_dict(), best_encoder_dir / "pytorch_model.bin")

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
            log.info("  New best val loss: %.5f  (epoch %d)  -> %s", best_val_loss, epoch, out_dir)

    log.info("EAGLE fine-tuning complete. Best val total loss: %.5f", best_val_loss)


if __name__ == "__main__":
    main()
