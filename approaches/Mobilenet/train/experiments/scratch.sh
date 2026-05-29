#!/bin/bash
#SBATCH --job-name=mobilenet_scratch
#SBATCH --time=08:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/mobilenet_scratch_%j.out
#SBATCH --error=logs/mobilenet_scratch_%j.err
#
# Experiment 1: from-scratch baseline
#   - random-init MobileNetV3 (no ImageNet pretrain)
#   - dirty (original) LiDAR GT
#   - silog only
#   - no ASPP, no EAGLE, no VN
# The absolute floor for what the architecture can learn.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
TEST_DIR="${TEST_DIR:-/cluster/courses/cil/monocular-depth-estimation/test}"
PROJ_DIR="${PROJ_DIR:-approaches/Mobilenet/results/scratch}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$(dirname "$PROJ_DIR")"

# Auto-resume so this script is safe to re-launch (chain-on-timeout pattern).
RESUME_FLAG=""
if [[ -f "$PROJ_DIR/checkpoints/latest.pth" ]]; then
    RESUME_FLAG="--resume $PROJ_DIR/checkpoints/latest.pth"
    echo "==> resuming from $PROJ_DIR/checkpoints/latest.pth"
fi

python3 -m approaches.Mobilenet.train.train \
    $RESUME_FLAG \
    --proj_dir   "$PROJ_DIR" \
    --train_dir  "$TRAIN_DIR" \
    --test_dir   "$TEST_DIR" \
    --image_size 576 \
    --epochs 20 \
    --batch_size 8 \
    --num_workers 8 \
    --lr 3e-4 \
    --warmup_steps 700 \
    --silog_weight 1.0
