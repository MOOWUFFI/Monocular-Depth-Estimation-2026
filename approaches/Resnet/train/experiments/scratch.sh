#!/bin/bash
#SBATCH --job-name=resnet_scratch
#SBATCH --time=08:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/resnet_scratch_%j.out
#SBATCH --error=logs/resnet_scratch_%j.err
#
# Experiment 1: from-scratch floor.
#   - random-init ResNet34 (no ImageNet pretrain)
#   - dirty (original) LiDAR GT
#   - silog only
#   - no ASPP, no EAGLE, no VN
# The absolute floor for what the architecture can learn.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
OUTPUT_DIR="${OUTPUT_DIR:-approaches/Resnet/results/scratch}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$(dirname "$OUTPUT_DIR")"

python3 -m approaches.Resnet.train.train \
    --from_scratch \
    --no_pretrained_encoder \
    --unfreeze_encoder \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 10 \
    --silog_weight 1.0 \
    --vnl_weight 0.0
