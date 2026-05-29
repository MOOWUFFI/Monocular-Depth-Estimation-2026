# claude.md — working notes for this repository

This file is the memory of the most important decisions and constraints for this
repository. Read it before making changes.

## What this repository is

This is a **self-contained repository**: it is meant to be copied/cloned on its
own. Two consequences that must always hold:

- **Complete isolation.** Nothing here may import from, or hard-code a path to,
  anything outside this repository, and nothing outside references anything in
  here. The repository root is the import root, so `scripts` and `approaches`
  are the top-level importable packages.
- **No personal names.** No team-member personal names may appear in any file,
  comment, docstring, path, or filename anywhere in the codebase. Keep it that
  way when adding code.

## Hard rules / decisions taken

1. **Branch:** development happens directly on `main`.
2. **Four approaches**, one per backbone, each self-contained under
   `approaches/`: `Mobilenet`, `Segformer`, `Mobilevit`, `Resnet`.
   - `Mobilenet` is the **chosen approach** (MobileNetV3-Small + ASPP + EAGLE +
     virtual-normal loss) and is the only one that ships trained checkpoints.
3. **Benchmarks (`benchmarks/`) are deferred** — the 3-benchmark code
   (NYUv2 / KITTI / ETH3D) will be added later. For now `benchmarks/` carries a
   placeholder README only.
4. **MobileNet binaries are kept**: per-epoch + best `.pth` checkpoints and the
   pointcloud `.html` files (from the top-line `base_aspp_eagle_vn` run). Other
   approaches keep an empty `results/pointclouds/` (no checkpoints) until they
   have their own.
5. **Verification is static only** — this environment has no `torch`/`cv2`. We
   verify with `python -m py_compile`, a structural import-graph check, and grep
   sweeps for personal names / stale references. We do *not* claim numerical
   parity was re-run; the porting preserves the original logic verbatim apart
   from import paths, renames, and name scrubbing.

## Folder structure

```
.
├── README.md                 # project overview + how to reproduce
├── claude.md                 # this file
├── requirements.txt
├── scripts/                  # approach-INDEPENDENT shared code
│   ├── constants.py          # dataset / benchmark paths, ImageNet stats
│   ├── losses.py             # silog, virtual_normal, eigen_cluster, within_cluster
│   ├── metrics.py            # siRMSE + standard depth metrics
│   ├── pointcloud.py         # depth.npy + rgb -> interactive 3D HTML
│   ├── visualize_depth.py    # depth.npy -> 2D PNG
│   └── clean_gt.py           # DROR de-noise of LiDAR GT (bulk, parallel)
├── benchmarks/               # NYUv2 / KITTI / ETH3D (DEFERRED — placeholder)
└── approaches/
    └── <Approach>/           # Mobilenet | Segformer | Mobilevit | Resnet
        ├── README.md         # which experiments this approach runs
        ├── model.py          # the network definition (+ build helper)
        ├── eagle.py          # EigenAggregationModule (model-side EAGLE)
        ├── results/
        │   ├── pointclouds/  # prediction pointcloud .html (MobileNet only so far)
        │   └── checkpoints/  # per-epoch .pth (MobileNet only)
        ├── train/            # training code; imports scripts.{constants,losses,metrics}
        └── inference/        # inference / submission code
```

## Import conventions

- Shared code: `from scripts.constants import TRAIN_DIR, TEST_DIR, ...`,
  `from scripts.losses import silog_loss, virtual_normal_loss,
  eigen_cluster_loss, within_cluster_consistency_loss`,
  `from scripts.metrics import all_metrics`.
- Within an approach: `from approaches.<Approach>.model import ...`,
  `from approaches.<Approach>.eagle import EigenAggregationModule`.
- Use **absolute** imports rooted at `scripts` / `approaches` (no relative
  `from .x import`). Entrypoints run as `python -m approaches.<Approach>.train.train`.
- Loss building blocks live in `scripts/losses.py`. Each approach's **loss
  wrapper** (which weights the terms) lives in that approach's `train/` because
  the wrappers differ (e.g. one weights log-depth, another linear depth).
  Schedulers also differ per approach and stay in each approach's `train/`.
- `EigenAggregationModule` is a model component that differs across backbones,
  so it is *not* shared — each approach carries its own `eagle.py`.

## Paths / constants

All dataset paths are centralised in `scripts/constants.py` and overridable via
environment variables (`CIL_DATA_ROOT`, `CIL_TRAIN_DIR`, `CIL_TEST_DIR`,
`CIL_CLEANED_GT_DIR`, `BENCHMARK_DATA_DIR`, `BENCHMARK_RESULTS_DIR`). Defaults
point at the course cluster dataset. No machine- or user-specific scratch paths
are hard-coded anywhere.
