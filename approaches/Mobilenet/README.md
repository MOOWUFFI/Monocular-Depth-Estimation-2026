# MobileNet approach (chosen approach)

MobileNetV3-Small encoder + (optional) ASPP-lite bottleneck + U-Net decoder,
with multi-stride **EAGLE** EiCue heads and a **Virtual Normal** geometric loss.
This is the project's chosen approach — the EAGLE-on-a-convolutional-backbone
result from the paper. The encoder stem is widened to 5 channels (RGB + X/Y
coordinate grids); the network predicts log-depth and is scored by per-image
siRMSE (identical to the leaderboard metric).

## Experiments

One training entrypoint (`train/train.py`) drives a five-experiment ablation;
each `train/experiments/*.sh` sets a different combination of flags:

| Experiment | GT | Pretrain | ASPP | EAGLE | VN | Losses |
|---|---|---|---|---|---|---|
| `scratch` | dirty | no | no | no | no | silog |
| `base` | dirty | yes | no | no | no | silog |
| `base_aspp` | cleaned | yes | yes | no | no | silog |
| `base_aspp_eagle` | cleaned | yes | yes | [32, 16] | no | silog + eig + within |
| `base_aspp_eagle_vn` | cleaned | yes | yes | [32, 16] | yes | silog + eig + within + VN |

Each experiment isolates one component, except `base → base_aspp`, which adds
ASPP *and* switches from dirty to DROR-cleaned GT at the same time.

The committed `results/checkpoints/` and `results/pointclouds/` belong to the
top-line `base_aspp_eagle_vn` run.

## Loss weights

- `silog_weight 1.0` — per-image siRMSE on sparse GT, identical to the metric.
- `eig_cluster_weight 0.05` — EAGLE Eq. 1; drives confident cluster assignment.
- `within_cluster_weight 0.1` — within-cluster variance of log-depth.
- `virtual_normal_weight 5.0`, `n_triplets 1024` — Yin et al. (ICCV 2019)
  3-point back-projected triangle-normal prior. Active only in `*_vn`.

The shared loss *building blocks* live in `scripts/losses.py`; the weighting
wrapper is `train/losses.py`. Dataset paths come from `scripts/constants.py`.

## Running (from the repository root)

```bash
# one experiment locally
bash approaches/Mobilenet/train/experiments/base_aspp_eagle_vn.sh

# or submit all five via slurm (idempotent, auto-resumes)
bash approaches/Mobilenet/train/experiments/run_all.sh
```

Each run writes to `results/<experiment>/`:
`checkpoints/` (best + per-epoch), `history.json`, `args.json`,
`predictions/`, `submission.csv`, `viz/` (2D panels + pointcloud HTML + EAGLE
panels), and `figures/` (loss + metric curves). The end-of-training eval runs
automatically; pass `--no_run_eval` to skip it.

## Inference

```bash
python -m approaches.Mobilenet.inference.inference \
    --ckpt approaches/Mobilenet/results/checkpoints/best.pth \
    --out_dir approaches/Mobilenet/results/predictions \
    --out_csv approaches/Mobilenet/results/submission.csv

# EAGLE eigenvector / cluster visualisations
python -m approaches.Mobilenet.inference.visualize_eagle \
    --ckpt approaches/Mobilenet/results/checkpoints/best.pth \
    --out_dir approaches/Mobilenet/results/eagle_viz

# interactive 3D pointcloud from any depth .npy + RGB
python -m scripts.pointcloud depth.npy rgb.png --out cloud.html
```
