"""Training entrypoint for the MobileNet approach.

Five experiments share this one script — each `experiments/*.sh` invokes it
with a different combination of flags:

    scratch            : silog | dirty GT | no pretrain  | no ASPP
    base               : silog | dirty GT | pretrained   | no ASPP
    base_aspp          : silog | cleaned  | pretrained   | ASPP
    base_aspp_eagle    : silog + eig + within | cleaned  | pretrained | ASPP + EAM[32,16]
    base_aspp_eagle_vn : + VN                  | cleaned  | pretrained | ASPP + EAM[32,16]

At end of training, when `--run_eval` is set (default true), the script runs
inference on `--test_dir`, writes the submission CSV, renders 2D + pointcloud +
EAGLE viz panels for a few validation IDs, and plots the loss curves from
`history.json`. Pass `--no_run_eval` to skip.

Usage (from the repository root):
    python -m approaches.Mobilenet.train.train \
        --proj_dir approaches/Mobilenet/results/base_aspp_eagle_vn \
        --use_aspp --eam_scales 32 16 --pretrained_encoder \
        --silog_weight 1.0 --eig_cluster_weight 0.05 --within_cluster_weight 0.1 \
        --virtual_normal_weight 5.0 --virtual_normal_n_triplets 1024 \
        --epochs 20 --batch_size 8 --lr 3e-4 --warmup_steps 700
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from scripts.constants import CLEANED_GT_DIR, TEST_DIR, TRAIN_DIR
from scripts.metrics import all_metrics
from approaches.Mobilenet.model import TinyDepthUNet, build_model
from approaches.Mobilenet.train.dataset import SparseDepthDataset, get_train_val_splits
from approaches.Mobilenet.train.losses import TotalLoss
from approaches.Mobilenet.train.schedulers import warmup_poly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--proj_dir", required=True, type=str,
                   help="Where to write checkpoints, history.json, viz.")
    p.add_argument("--train_dir", type=str, default=TRAIN_DIR,
                   help="Source dir with <basename>_{rgb.png,depth.npy}.")
    p.add_argument("--test_dir", type=str, default=TEST_DIR,
                   help="Test-set RGB dir for end-of-training submission.")
    p.add_argument("--gt_depth_dir", type=str, default=None,
                   help="Override GT depth source (used for cleaned-GT runs). "
                        "When set, RGB still loads from --train_dir.")
    # Model
    p.add_argument("--use_aspp", action="store_true",
                   help="Enable ASPP-lite bottleneck. Off = plain 1x1 projection.")
    p.add_argument("--pretrained_encoder", action="store_true",
                   help="Initialise MobileNetV3-Small from ImageNet weights.")
    p.add_argument("--bottleneck_channels", type=int, default=64)
    p.add_argument("--decoder_channels", type=int, nargs=4, default=[64, 48, 32, 16])
    p.add_argument("--eam_scales", type=int, nargs="*", default=[],
                   help="EAM strides. Empty = no EAGLE. Supported: 32, 16, 8.")
    p.add_argument("--eam_k", type=int, default=4,
                   help="Number of leading non-trivial eigenvectors kept.")
    p.add_argument("--num_clusters", type=int, default=10,
                   help="EAM cluster centers.")
    p.add_argument("--eam_sigma_color", type=float, default=0.2)
    # Losses
    p.add_argument("--silog_weight", type=float, default=1.0)
    p.add_argument("--eig_cluster_weight", type=float, default=0.0)
    p.add_argument("--within_cluster_weight", type=float, default=0.0)
    p.add_argument("--virtual_normal_weight", type=float, default=0.0)
    p.add_argument("--virtual_normal_fov_deg", type=float, default=60.0)
    p.add_argument("--virtual_normal_n_triplets", type=int, default=1024)
    # Data
    p.add_argument("--image_size", type=int, default=576)
    p.add_argument("--val_count", type=int, default=95)
    # Train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=700)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    # Eval
    p.add_argument("--run_eval", action=argparse.BooleanOptionalAction, default=True,
                   help="At end of training, run inference + submission + viz. "
                        "Pass --no_run_eval to skip.")
    p.add_argument("--n_viz", type=int, default=8,
                   help="How many sample IDs to render viz panels for.")
    return p.parse_args()


def _setup_proj(proj_dir: Path, args: argparse.Namespace) -> None:
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "checkpoints").mkdir(exist_ok=True)
    with open(proj_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


def _build_model(args: argparse.Namespace, device: torch.device) -> TinyDepthUNet:
    return build_model(vars(args), device)


@torch.no_grad()
def _validate(model: TinyDepthUNet, loader: DataLoader, loss_fn: TotalLoss,
              device: torch.device) -> dict:
    model.eval()
    agg = {"total": 0.0, "silog": 0.0, "eig_cluster": 0.0,
           "within_cluster": 0.0, "virtual_normal": 0.0}
    metrics_sum: dict[str, list[float]] = {}
    n_batches = 0
    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        with torch.amp.autocast("cuda"):
            out = model(rgb)
            _, comps = loss_fn(out, depth, mask)
        for k in agg:
            agg[k] += float(comps.get(k, 0.0))
        m = all_metrics(torch.exp(out["log_depth"]), depth, mask)
        for k, v in m.items():
            v = float(v.item())
            if np.isfinite(v):
                metrics_sum.setdefault(k, []).append(v)
        n_batches += 1
    for k in agg:
        agg[k] /= max(1, n_batches)
    metrics_mean = {k: float(np.mean(vs)) if vs else float("nan")
                    for k, vs in metrics_sum.items()}
    return {"loss": agg, "metrics": metrics_mean}


def _save_ckpt(path: Path, model, optimizer, scheduler, scaler, epoch: int,
               best_val: float, args: argparse.Namespace) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val": best_val,
        "args": vars(args),
    }, path)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    proj = Path(args.proj_dir).expanduser().resolve()
    _setup_proj(proj, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_files, val_files = get_train_val_splits(args.train_dir, args.val_count)
    print(f"train: {len(train_files)} | val: {len(val_files)}", flush=True)

    train_ds = SparseDepthDataset(args.train_dir, train_files, is_train=True,
                                   image_size=args.image_size,
                                   gt_depth_dir=args.gt_depth_dir)
    val_ds = SparseDepthDataset(args.train_dir, val_files, is_train=False,
                                 image_size=args.image_size,
                                 gt_depth_dir=args.gt_depth_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = _build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {n_params:.2f} M params | eam_scales={model.eam_scales} "
          f"| use_aspp={model.use_aspp}", flush=True)

    loss_fn = TotalLoss(
        silog_weight=args.silog_weight,
        eig_cluster_weight=args.eig_cluster_weight,
        within_cluster_weight=args.within_cluster_weight,
        virtual_normal_weight=args.virtual_normal_weight,
        virtual_normal_fov_deg=args.virtual_normal_fov_deg,
        virtual_normal_n_triplets=args.virtual_normal_n_triplets,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, args.encoder_lr_mult, args.weight_decay),
        betas=(0.9, 0.999),
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = warmup_poly(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        print(f"resuming from {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))

    history: list[dict] = []
    history_path = proj / "history.json"
    # When resuming, carry forward the existing history so the file
    # accumulates over restarts instead of being overwritten each time.
    # Drop any entries past the resumed epoch — those would belong to a
    # later state of training that we don't have a checkpoint for.
    if args.resume and history_path.exists():
        try:
            with open(history_path) as f:
                prior = json.load(f).get("epochs", [])
            history = [e for e in prior if int(e.get("epoch", 0)) <= start_epoch]
            print(f"carried over {len(history)} prior epochs from history.json",
                  flush=True)
        except Exception as e:
            print(f"warning: could not load prior history.json: {e}", flush=True)
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        agg = {"total": 0.0, "silog": 0.0, "eig_cluster": 0.0,
               "within_cluster": 0.0, "virtual_normal": 0.0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}",
                    leave=False, file=sys.stdout)
        for rgb, depth, mask in pbar:
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(rgb)
                loss, comps = loss_fn(out, depth, mask)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            for k in agg:
                agg[k] += float(comps.get(k, 0.0))
            pbar.set_postfix(loss=f"{comps['total']:.3f}")
        for k in agg:
            agg[k] /= max(1, len(train_loader))

        val = _validate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"epoch {epoch+1:3d}/{args.epochs}  "
            f"train_total={agg['total']:.4f}  "
            f"val_total={val['loss']['total']:.4f}  "
            f"val_sirmse={val['metrics'].get('sirmse', float('nan')):.4f}  "
            f"lr={lr_now:.2e}  elapsed={elapsed/60:.1f}m",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train": agg,
            "val_loss": val["loss"],
            "val_metrics": val["metrics"],
            "lr": lr_now,
        })
        with open(history_path, "w") as f:
            json.dump({"epochs": history, "args": vars(args)}, f, indent=2)

        ckpt_dir = proj / "checkpoints"
        _save_ckpt(ckpt_dir / "latest.pth", model, optimizer, scheduler, scaler,
                   epoch, best_val, args)
        _save_ckpt(ckpt_dir / f"epoch_{epoch+1:03d}.pth", model, optimizer, scheduler,
                   scaler, epoch, best_val, args)
        val_metric = val["metrics"].get("sirmse", float("inf"))
        if np.isfinite(val_metric) and val_metric < best_val:
            best_val = val_metric
            _save_ckpt(ckpt_dir / "best.pth", model, optimizer, scheduler, scaler,
                       epoch, best_val, args)

    print(f"training done. best val_sirmse={best_val:.4f}", flush=True)

    if args.run_eval:
        run_end_of_training_eval(proj, args, device)


@torch.no_grad()
def run_end_of_training_eval(proj: Path, args: argparse.Namespace,
                             device: torch.device) -> None:
    """Inference on test set + submission CSV + viz panels + loss curves."""
    print("=== end-of-training eval ===", flush=True)
    ckpt_path = proj / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        ckpt_path = proj / "checkpoints" / "latest.pth"
    print(f"loading {ckpt_path}", flush=True)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model(args, device).eval()
    model.load_state_dict(ck["model_state_dict"])

    pred_dir = proj / "predictions"
    pred_dir.mkdir(exist_ok=True)
    _run_inference(model, args.test_dir, pred_dir, args.image_size, device)

    # Submission CSV.
    csv_out = proj / "submission.csv"
    _make_submission_csv(pred_dir, csv_out)
    print(f"wrote {csv_out}", flush=True)

    # Viz panels on the first n_viz training samples (we have GT here).
    sample_basenames = sorted([
        f.replace("_rgb.png", "")
        for f in os.listdir(args.train_dir) if f.endswith("_rgb.png")
    ])[: args.n_viz]
    viz_dir = proj / "viz"
    viz_dir.mkdir(exist_ok=True)
    for base in sample_basenames:
        _viz_one_sample(model, args, base, viz_dir, device)
    print(f"wrote {args.n_viz} viz panels to {viz_dir}", flush=True)

    # Loss curves.
    try:
        subprocess.run(
            [sys.executable, "-m", "approaches.Mobilenet.train.plot_metrics",
             "--proj_dir", str(proj)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"plot_metrics failed: {e}", flush=True)


@torch.no_grad()
def _run_inference(model, test_dir: str, out_dir: Path, image_size: int,
                   device: torch.device) -> None:
    """Predict depth for every *_rgb.png in test_dir; save as .npy at (560, 560)."""
    files = sorted(f for f in os.listdir(test_dir) if f.endswith("_rgb.png"))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    print(f"inference on {len(files)} test images", flush=True)
    for f in tqdm(files, desc="infer", file=sys.stdout):
        img = Image.open(os.path.join(test_dir, f)).convert("RGB")
        img_t = norm(
            transforms.functional.to_tensor(img.resize((image_size, image_size), Image.BILINEAR))
        ).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda"):
            out = model(img_t)
        depth = torch.exp(out["log_depth"]).squeeze().float().cpu().numpy()
        # Resize to 560 for the submission format.
        import torch.nn.functional as F
        depth_t = torch.from_numpy(depth)[None, None]
        depth_560 = F.interpolate(depth_t, size=(560, 560), mode="bilinear",
                                  align_corners=False).squeeze().numpy()
        base = f.replace("_rgb.png", "")
        np.save(out_dir / f"{base}_pred_depth.npy", depth_560.astype(np.float32))


def _make_submission_csv(pred_dir: Path, csv_out: Path) -> None:
    import base64
    import zlib
    import pandas as pd

    rows = []
    for p in sorted(pred_dir.glob("*_pred_depth.npy")):
        depth = np.load(p)
        depth = np.nan_to_num(depth, nan=100.0, posinf=100.0, neginf=0.0)
        base = p.stem.replace("_pred_depth", "")
        # base looks like "test_000123"; the leaderboard expects "test_000123_depth"
        encoded = base64.b64encode(
            zlib.compress(np.asarray(depth, dtype=np.float16).tobytes(), level=9)
        ).decode("utf-8")
        rows.append({"id": f"{base}_depth", "Depths": encoded})
    pd.DataFrame(rows, columns=["id", "Depths"]).to_csv(csv_out, index=False)


@torch.no_grad()
def _viz_one_sample(model, args: argparse.Namespace, base: str, viz_dir: Path,
                    device: torch.device) -> None:
    """Save a per-sample 2D panel (RGB | GT | pred | error) and a pointcloud HTML.
    For EAGLE runs, also save the EAGLE viz (eigvecs + cluster map)."""
    import matplotlib.pyplot as plt

    rgb_path = os.path.join(args.train_dir, f"{base}_rgb.png")
    gt_path = os.path.join(args.gt_depth_dir or args.train_dir, f"{base}_depth.npy")
    if not os.path.exists(rgb_path):
        return

    img = Image.open(rgb_path).convert("RGB")
    rgb_resized = img.resize((args.image_size, args.image_size), Image.BILINEAR)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rgb_t = norm(transforms.functional.to_tensor(rgb_resized)).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda"):
        out = model(rgb_t)
    pred = torch.exp(out["log_depth"]).squeeze().float().cpu().numpy()

    gt = None
    if os.path.exists(gt_path):
        gt_raw = np.load(gt_path)
        gt = np.array(Image.fromarray(gt_raw).resize(
            (args.image_size, args.image_size), Image.NEAREST
        ))

    # 2D panel.
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(np.asarray(rgb_resized)); ax[0].set_title("RGB"); ax[0].axis("off")

    pred_log = np.log(np.clip(pred, 1e-3, None))
    if gt is not None:
        valid = gt > 1e-3
        gt_log = np.where(valid, np.log(np.clip(gt, 1e-3, None)), np.nan)
    else:
        valid = None
        gt_log = None

    # Common log-vis range covering BOTH gt valid pixels and pred — using
    # gt's range alone would map a degenerate pred (e.g. uniform 1.27m) to
    # the very bottom of magma, rendering pred as a black image.
    samples = [pred_log.ravel()]
    if gt_log is not None and valid.any():
        samples.append(gt_log[valid].ravel())
    lo, hi = np.percentile(np.concatenate(samples), [2, 98])
    if hi - lo < 1e-3:
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5

    if gt is not None:
        ax[1].imshow(gt_log, cmap="magma", vmin=lo, vmax=hi)
        ax[1].set_title(f"GT ({100*valid.mean():.1f}% valid)")
    else:
        ax[1].text(0.5, 0.5, "no GT", ha="center", va="center", transform=ax[1].transAxes)
    ax[1].axis("off")

    ax[2].imshow(pred_log, cmap="magma", vmin=lo, vmax=hi)
    ax[2].set_title(f"pred  med={np.median(pred):.2f}m"); ax[2].axis("off")
    if gt is not None:
        diff = np.where(valid, pred_log - gt_log, np.nan)
        lim = max(1e-3, np.nanpercentile(np.abs(diff), 98))
        ax[3].imshow(diff, cmap="seismic", vmin=-lim, vmax=lim)
        ax[3].set_title(f"log(pred) - log(GT)  ±{lim:.2f}")
    ax[3].axis("off")
    fig.tight_layout()
    fig.savefig(viz_dir / f"{base}_panel.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Pointcloud HTML.
    try:
        np.save(viz_dir / f"{base}_pred_depth.npy", pred.astype(np.float32))
        rgb_save = viz_dir / f"{base}_rgb.png"
        rgb_resized.save(rgb_save)
        subprocess.run(
            [sys.executable, "-m", "scripts.pointcloud",
             str(viz_dir / f"{base}_pred_depth.npy"),
             str(rgb_save),
             "--out", str(viz_dir / f"{base}_pcd.html")],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  pointcloud {base}: {e}", flush=True)

    # EAGLE viz: eigvecs + cluster map per scale.
    if out["Us"]:
        _viz_eagle_panel(out, np.asarray(rgb_resized), viz_dir / f"{base}_eagle.png")


def _viz_eagle_panel(out: dict, rgb_arr: np.ndarray, save_path: Path) -> None:
    """One row per active EAM scale: RGB | clusters | u1 | u2 | u3 | u4."""
    import matplotlib.pyplot as plt

    scales = sorted(out["Us"].keys(), reverse=True)
    cols = 6
    fig, axes = plt.subplots(len(scales), cols, figsize=(2.5 * cols, 2.5 * len(scales)))
    if len(scales) == 1:
        axes = np.array([axes])
    for r, s in enumerate(scales):
        U = out["Us"][s][0].float().cpu().numpy()
        logits = out["logits"][s][0].float().cpu().numpy()
        N, k_total = U.shape
        h = w = int(round(np.sqrt(N)))
        cluster_id = logits.argmax(axis=-1).reshape(h, w)
        axes[r, 0].imshow(rgb_arr); axes[r, 0].set_title(f"RGB"); axes[r, 0].axis("off")
        axes[r, 1].imshow(cluster_id, cmap="tab20", interpolation="nearest")
        axes[r, 1].set_title(f"clusters s={s}"); axes[r, 1].axis("off")
        for i in range(4):
            if i < k_total:
                axes[r, 2 + i].imshow(U[:, i].reshape(h, w), cmap="coolwarm")
                axes[r, 2 + i].set_title(f"u_{i+1}")
            axes[r, 2 + i].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
