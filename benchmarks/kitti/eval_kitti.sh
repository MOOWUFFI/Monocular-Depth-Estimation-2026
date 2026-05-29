#!/bin/bash
#SBATCH --job-name=kitti_eval
#SBATCH --time=00:30:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=4
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/kitti_eval_%j.out
#SBATCH --error=logs/kitti_eval_%j.err
#
# Cross-dataset generalisation: evaluate the trained MobileNet experiments on
# the Marigold-curated KITTI Eigen test split (652 pairs).
#
# One-time data setup (run once on the login node, not via this sbatch):
#   mkdir -p data/benchmarks/kitti
#   wget -P data/benchmarks/kitti \
#       https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti/kitti_eigen_split_test.tar
#   tar -xf data/benchmarks/kitti/kitti_eigen_split_test.tar -C data/benchmarks/kitti/

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

KITTI_DIR="${KITTI_DIR:-${BENCHMARK_DATA_DIR:-data/benchmarks}/kitti}"
CKPT_ROOT="${CKPT_ROOT:-approaches/Mobilenet/results}"
OUT_JSON="${OUT_JSON:-results/benchmarks/kitti_eval.json}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$(dirname "$OUT_JSON")"

python3 -m benchmarks.kitti.inference \
    --data_root "$KITTI_DIR" \
    --ckpt_root "$CKPT_ROOT" \
    --out_json  "$OUT_JSON"
