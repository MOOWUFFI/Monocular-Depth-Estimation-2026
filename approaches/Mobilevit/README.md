# MobileViT approach

MobileViT-XX-Small encoder + all-MLP `DepthDecodeHead` (metric depth) with
**EAGLE** EiCue heads (`EigenAggregationModule`) attached as side channels at
the small-grid encoder stages. The EAMs do **not** modify the features fed to
the decoder; their eigenvector-derived cluster logits feed two auxiliary
losses (EAGLE Eq. 1 and a within-cluster depth-consistency term). The network
predicts positive metric depth (Softplus) and is supervised by SILog.

MobileViT-XX-Small has 5 encoder stages
(`hidden_sizes = neck_hidden_sizes[1:-1] = [32, 64, 96, 128, 160]`). EAMs are
recommended at the small-grid stages 3 (16x16) and 4 (8x8); earlier stages
produce too many patches for the N x N affinity. Unlike SegFormer, the input is
raw `[0, 1]` (no ImageNet normalization).

## Experiments

One training entrypoint (`train/train.py`) drives the ablation; each
`train/experiments/*.sh` sets a different combination of flags:

| Experiment | Init | EAM stages | Losses |
|---|---|---|---|
| `eagle_scratch` | pretrained MobileViT-XX-Small, random head (`--from_scratch`) | 3, 4 | silog + eig + within |
| `eagle_eam3s4` | Stage-2 distilled encoder + head (`--stage2_encoder/--stage2_head`) | 3, 4 | silog + eig + within |

The `eagle_eam3s4` script expects `STAGE2_ENCODER` / `STAGE2_HEAD` env vars
pointing at a Stage-2 distillation run's `best_encoder` directory and
`best_depth_head.pt`.

## Loss weights

- `silog_weight 1.0` — scale-invariant log loss on metric depth (= leaderboard metric).
- `eig_cluster_weight 0.05` — EAGLE Eq. 1; drives confident cluster assignment.
- `within_cluster_weight 0.1` — soft-assignment-weighted within-cluster variance of depth.

The shared loss *building blocks* live in `scripts/losses.py`; the weighting
wrapper is `train/losses.py` (`EagleDepthLoss`). Dataset paths come from
`scripts/constants.py` (override via `CIL_TRAIN_DIR` / `CIL_TEST_DIR`).

## Running (from the repository root)

```bash
# from scratch
bash approaches/Mobilevit/train/experiments/eagle_scratch.sh

# from a Stage-2 distilled checkpoint
STAGE2_ENCODER=/path/to/best_encoder \
STAGE2_HEAD=/path/to/best_depth_head.pt \
bash approaches/Mobilevit/train/experiments/eagle_eam3s4.sh
```

Each run writes to `results/<experiment>/<run_tag>/`:
`best_checkpoint.pt`, `latest_checkpoint.pt`, `best_encoder/`,
`best_depth_head.pt`, `best_eams.pt`, `history.json`, `run_config.json`.

## Inference

```bash
python -m approaches.Mobilevit.inference.inference \
    --ckpt approaches/Mobilevit/results/<experiment>/<run_tag>/best_checkpoint.pt \
    --out_dir approaches/Mobilevit/results/<experiment>/<run_tag>/predictions \
    --out_csv approaches/Mobilevit/results/<experiment>/<run_tag>/submission.csv

# interactive 3D pointcloud from any depth .npy + RGB
python -m scripts.pointcloud depth.npy rgb.png --out cloud.html
```
