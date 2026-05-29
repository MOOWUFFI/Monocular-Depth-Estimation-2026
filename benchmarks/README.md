# benchmarks

Cross-dataset **generalisation** benchmarks for the trained depth models. After
training on the course dataset, we evaluate the checkpoints zero-shot on three
established depth datasets, each with its own capture modality and evaluation
protocol. Every benchmark loads a trained checkpoint, runs inference on that
dataset's test split, and reports the standard depth metrics under per-image
affine (scale + shift) alignment, matching common monocular-depth evaluation.

Benchmark data and checkpoint roots come from `scripts/constants.py`
(`BENCHMARK_DATA_DIR`, `BENCHMARK_RESULTS_DIR`) and are overridable via
environment variables or each script's command-line flags.

## The three benchmarks

### NYU-Depth-v2 (`nyuv2/`)
Indoor scenes captured with a Microsoft Kinect (structured-light RGB-D), with
dense depth up to ~10 m. We evaluate on the standard **Eigen test split** with
the **Eigen crop**, reporting SILog, AbsRel, and δ1. The evaluator is
self-contained — it defines every model architecture inline so it can score any
of the project's checkpoints (ResNet34+EAGLE, SegFormer, MobileViT, MobileNet,
and the standalone UNet/ResNet baselines).

```bash
python -m benchmarks.nyuv2.inference --data_dir data/benchmarks/nyuv2
```

### ETH3D (`eth3d/`)
High-resolution outdoor + indoor scenes with **laser-scanned** ground truth,
the most accurate of the three. We follow the Depth Anything V2 protocol and
report SI-RMSE, AbsRel, and δ1 over `bench_*_rgb.png` / `bench_*_depth.npy`
pairs, with either per-image affine alignment or a single global-median scale.

```bash
python -m benchmarks.eth3d.inference --data_dir data/benchmarks/eth3d \
    --checkpoint_dirs <ckpt_dir_or_file> [...]
```

### KITTI (`kitti/`)
Outdoor driving scenes with **sparse LiDAR** ground truth, evaluated on the
Marigold-curated Eigen test split under the Garg crop with the
`1e-3 < gt < 80 m` mask and per-image median scaling. Reports SILog, AbsRel,
and δ1 for the trained MobileNet experiments.

```bash
# one-time: download + unpack the Marigold KITTI Eigen split into data/benchmarks/kitti
python -m benchmarks.kitti.inference \
    --data_root data/benchmarks/kitti --ckpt_root approaches/Mobilenet/results
```

## Reported metrics

- **SI-RMSE / SILog** — scale-invariant RMSE in log space.
- **AbsRel** — mean absolute relative error after alignment.
- **δ1** — fraction of pixels with max(pred/gt, gt/pred) < 1.25 (a value in
  [0, 1]; multiply by 100 for a percentage).
