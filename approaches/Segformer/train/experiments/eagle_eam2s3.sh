#!/bin/bash
#SBATCH --job-name=segformer_eagle_eam2s3
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=4
#SBATCH --output=logs/segformer_eagle_eam2s3_%j.out
#SBATCH --error=logs/segformer_eagle_eam2s3_%j.err
#
# SegFormer-B0 + EAGLE from a Stage-2 distilled checkpoint:
#   - encoder + depth head loaded from a Stage-2 distillation run
#   - EAMs at stages 2 and 3
#   - silog + L_eig + L_within_cluster
# Set STAGE2_ENCODER / STAGE2_HEAD to the Stage-2 best_encoder dir and
# best_depth_head.pt produced by the distillation stage.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
PROJ_DIR="${PROJ_DIR:-approaches/Segformer/results/eagle_eam2s3}"
STAGE2_ENCODER="${STAGE2_ENCODER:-}"
STAGE2_HEAD="${STAGE2_HEAD:-}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$PROJ_DIR"

# Auto-resume so this script is safe to re-launch (chain-on-timeout pattern).
RESUME_FLAG=""
LATEST="$(ls -1 "$PROJ_DIR"/*/latest_checkpoint.pt 2>/dev/null | head -n1 || true)"
if [[ -n "$LATEST" ]]; then
    RESUME_FLAG="--resume_from $LATEST"
    echo "==> resuming from $LATEST"
fi

python3 -m approaches.Segformer.train.train \
    $RESUME_FLAG \
    --stage2_encoder "$STAGE2_ENCODER" \
    --stage2_head    "$STAGE2_HEAD" \
    --train_dir   "$TRAIN_DIR" \
    --output_dir  "$PROJ_DIR" \
    --eam_stages 2 3 \
    --eam_k 4 \
    --num_clusters 10 \
    --eam_sigma_color 0.2 \
    --input_size 256 \
    --num_epochs 30 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --silog_weight 1.0 \
    --eig_cluster_weight 0.05 \
    --within_cluster_weight 0.1
