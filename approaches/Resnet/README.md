# ResNet34 + EAGLE approach

ResNet34 encoder + a SegFormer-style `DepthDecodeHead` (optional ASPP on the
deepest feature map) + per-stage **EAGLE** EigenAggregationModules. The encoder
emits four hidden stages (C = 64/128/256/512); the head fuses all stages and
predicts depth via softplus. EAGLE EiCue heads attach at chosen encoder stages
and feed only the clustering losses — they never modify the decoder features.
Models are scored by per-image siRMSE (identical to the leaderboard metric).

## Experiments

One training entrypoint (`train/train.py`) drives a five-experiment ablation;
each `train/experiments/*.sh` sets a different combination of flags:

| Experiment | GT | Pretrain | ASPP | EAGLE | VN | Losses |
|---|---|---|---|---|---|---|
| `scratch` | dirty | no | no | no | no | silog |
| `base` | dirty | yes | no | no | no | silog |
| `base_aspp` | cleaned | yes | yes | no | no | silog |
| `base_aspp_eagle` | cleaned | yes | yes | stages [2, 3] | no | silog + eig + within |
| `base_aspp_eagle_vn` | cleaned | yes | yes | stages [2, 3] | yes | silog + eig + within + VN |

Each experiment isolates one component, except `base -> base_aspp`, which adds
ASPP *and* switches from dirty to DROR-cleaned GT at the same time. The encoder is
unfrozen in every run (`--unfreeze_encoder`, `encoder_lr_mult 0.1`).

## Loss weights

- `silog_weight 1.0` — per-image siRMSE on sparse GT, identical to the metric.
- `eig_cluster_weight 0.05` — EAGLE Eq. 1; drives confident cluster assignment.
- `within_cluster_weight 0.1` — within-cluster variance of predicted depth.
- `vnl_weight 5.0` — Yin et al. (ICCV 2019) 3-point back-projected
  triangle-normal prior. Active only in `*_vn`.

The shared loss *building blocks* live in `scripts/losses.py`; the weighting
wrapper is `train/losses.py` (`EagleDepthLoss`). The model-side EAGLE helpers
(affinity / Laplacian / `EigenAggregationModule`) live in `eagle.py`. Dataset
paths come from `scripts/constants.py`.

## Running (from the repository root)

```bash
# one experiment locally
LOCAL=1 bash approaches/Resnet/train/experiments/base_aspp_eagle_vn.sh

# or submit all five via slurm
bash approaches/Resnet/train/experiments/run_all.sh

# run a single config through the dispatcher
bash approaches/Resnet/train/experiments/run_all.sh base_aspp_eagle
```

Each run writes to `--output_dir/<run_tag>/`: `best_checkpoint.pt`,
`latest_checkpoint.pt`, `best_encoder/` (encoder dir), `best_depth_head.pt`,
`best_eams.pt`, `history.json`, and `run_config.json`. The checkpoint format
is intentionally split into an **encoder directory** + a **depth-head file** so
inference can reload them independently.

## Inference

Inference loads the saved encoder dir + depth-head file (not a single `.pth`),
predicts depth for every test RGB, writes per-image `.npy` + magma viz, and
builds the submission CSV.

```bash
# via the slurm wrapper (auto-discovers the latest base_aspp_eagle_vn run)
bash approaches/Resnet/train/experiments/infer.sh

# or directly
python -m approaches.Resnet.inference.inference \
    --output_dir approaches/Resnet/results/predictions \
    --output_csv submission.csv \
    --encoder_dir <run>/best_encoder \
    --head_path   <run>/best_depth_head.pt \
    --use_aspp

# interactive 3D pointcloud from any depth .npy + RGB
python -m scripts.pointcloud depth.npy rgb.png --out cloud.html
```

Pass `--use_aspp` only if the run was trained with ASPP (it must match).
