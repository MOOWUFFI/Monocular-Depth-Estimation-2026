"""Plot training curves from a run's history.json.

Reads `<proj_dir>/history.json` (written by train.py) and produces:

  <proj_dir>/figures/loss_total.png         train + val total loss per epoch
  <proj_dir>/figures/loss_components.png    per-component train losses
  <proj_dir>/figures/val_loss_components.png
  <proj_dir>/figures/val_metrics.png        siRMSE / RMSE / AbsRel curves
  <proj_dir>/figures/val_delta.png          d1 / d2 / d3 curves
  <proj_dir>/figures/lr_schedule.png

Usage:
    python -m approaches.Mobilenet.train.plot_metrics \
        --proj_dir approaches/Mobilenet/results/base_aspp_eagle_vn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(epochs: list, *keys, default=float("nan")):
    out = []
    for e in epochs:
        cur = e
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                cur = default
                break
            cur = cur[k]
        out.append(cur)
    return out


def _plot(xs, series_dict, ylabel, title, save_path, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys in series_dict.items():
        ys_arr = np.asarray(ys, dtype=float)
        if np.isfinite(ys_arr).any():
            ax.plot(xs, ys_arr, label=label, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj_dir", required=True)
    args = p.parse_args()

    proj = Path(args.proj_dir).expanduser()
    history_path = proj / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"no history.json under {proj}")
    with open(history_path) as f:
        hist = json.load(f)
    epochs = hist["epochs"]
    if not epochs:
        raise RuntimeError("history.json has no epoch records")

    fig_dir = proj / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    xs = [e["epoch"] for e in epochs]
    _plot(xs,
          {"train_total": _series(epochs, "train", "total"),
           "val_total":   _series(epochs, "val_loss", "total")},
          "loss", "Total loss", fig_dir / "loss_total.png")

    comp_keys = ["silog", "eig_cluster", "within_cluster", "virtual_normal"]
    _plot(xs, {k: _series(epochs, "train", k) for k in comp_keys},
          "loss", "Train loss components", fig_dir / "loss_components.png")
    _plot(xs, {k: _series(epochs, "val_loss", k) for k in comp_keys},
          "loss", "Val loss components", fig_dir / "val_loss_components.png")

    _plot(xs,
          {"sirmse": _series(epochs, "val_metrics", "sirmse"),
           "rmse":   _series(epochs, "val_metrics", "rmse"),
           "absrel": _series(epochs, "val_metrics", "absrel")},
          "metric", "Val metrics", fig_dir / "val_metrics.png")
    _plot(xs,
          {"d1": _series(epochs, "val_metrics", "d1"),
           "d2": _series(epochs, "val_metrics", "d2"),
           "d3": _series(epochs, "val_metrics", "d3")},
          "fraction", "Val delta accuracies", fig_dir / "val_delta.png")
    _plot(xs, {"lr": [e.get("lr", float("nan")) for e in epochs]},
          "lr", "Learning rate", fig_dir / "lr_schedule.png", log_y=True)

    print(f"wrote {len(list(fig_dir.glob('*.png')))} figures to {fig_dir}")


if __name__ == "__main__":
    main()
