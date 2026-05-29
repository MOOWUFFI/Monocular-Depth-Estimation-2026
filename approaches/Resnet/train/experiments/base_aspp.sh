#!/bin/bash
#SBATCH --job-name=resnet_base_aspp
#SBATCH --time=08:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/resnet_base_aspp_%j.out
#SBATCH --error=logs/resnet_base_aspp_%j.err
#
# Experiment 3: + ASPP + cleaned GT.
#   - ImageNet-pretrained ResNet34 encoder
#   - ASPP on the deepest feature map (multi-scale dilated context)
#   - cleaned (DROR-denoised) LiDAR GT
#   - silog only
# Two changes vs base: ASPP and cleaned GT, matching the pipeline progression.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
GT_DIR="${GT_DIR:-${CIL_CLEANED_GT_DIR:-./train_depth_clean}}"
OUTPUT_DIR="${OUTPUT_DIR:-approaches/Resnet/results/base_aspp}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$(dirname "$OUTPUT_DIR")"

python3 -m approaches.Resnet.train.train \
    --from_scratch \
    --pretrained_encoder \
    --unfreeze_encoder \
    --use_aspp \
    --train_dir "$TRAIN_DIR" \
    --gt_depth_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 10 \
    --silog_weight 1.0 \
    --vnl_weight 0.0
