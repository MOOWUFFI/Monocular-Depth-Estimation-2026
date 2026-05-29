"""Render EAGLE eigvecs + EiCue cluster maps for a checkpoint.

For each picked sample, saves a panel with one row per active EAM scale:
RGB | clusters | u_1 | u_2 | u_3 | u_4. Useful for diagnosing whether EAM
actually formed coherent structure or just bounced around.

    python -m approaches.Mobilenet.inference.visualize_eagle \
        --ckpt approaches/Mobilenet/results/checkpoints/best.pth \
        --test_dir /cluster/courses/cil/monocular-depth-estimation/train \
        --out_dir approaches/Mobilenet/results/eagle_viz \
        --n 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from scripts.constants import IMAGENET_MEAN, IMAGENET_STD, TRAIN_DIR
from approaches.Mobilenet.model import build_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_dir", default=TRAIN_DIR)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--image_size", type=int, default=None,
                   help="Defaults to the checkpoint's training image_size.")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = dict(ck.get("args", {}))
    train_args["pretrained_encoder"] = False  # weights come from the checkpoint
    image_size = args.image_size or int(train_args.get("image_size", 576))
    model = build_model(train_args, device).eval()
    model.load_state_dict(ck["model_state_dict"])
    if not model.eam_scales:
        raise SystemExit("checkpoint has no active EAM scales — nothing to visualise")
    print(f"eam_scales={model.eam_scales}  image_size={image_size}")

    files = sorted(f for f in os.listdir(args.test_dir) if f.endswith("_rgb.png"))
    step = max(1, len(files) // args.n)
    picks = files[::step][: args.n]
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    for fname in picks:
        base = fname.replace("_rgb.png", "")
        img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
        rgb_resized = img.resize((image_size, image_size), Image.BILINEAR)
        rgb_t = norm(transforms.functional.to_tensor(rgb_resized)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda"):
            out = model(rgb_t)

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
            axes[r, 0].imshow(np.asarray(rgb_resized)); axes[r, 0].set_title(f"RGB")
            axes[r, 0].axis("off")
            axes[r, 1].imshow(cluster_id, cmap="tab20", interpolation="nearest")
            axes[r, 1].set_title(f"clusters s={s}"); axes[r, 1].axis("off")
            for i in range(4):
                if i < k_total:
                    axes[r, 2 + i].imshow(U[:, i].reshape(h, w), cmap="coolwarm")
                    axes[r, 2 + i].set_title(f"u_{i+1}")
                axes[r, 2 + i].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{base}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  + {base}.png")
    print(f"done — {len(picks)} panels in {out_dir}")


if __name__ == "__main__":
    main()
