#!/bin/bash
#SBATCH --job-name=resnet_base_aspp_eagle_vn
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/resnet_base_aspp_eagle_vn_%j.out
#SBATCH --error=logs/resnet_base_aspp_eagle_vn_%j.err
#
# Experiment 5 (top-line): + Virtual Normal geometric loss.
#   - everything in base_aspp_eagle.sh
#   - + Virtual Normal (Yin et al. ICCV 2019), weight 5.0
# Isolates the contribution of the geometric VN prior on top of base_aspp_eagle.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
GT_DIR="${GT_DIR:-${CIL_CLEANED_GT_DIR:-./train_depth_clean}}"
OUTPUT_DIR="${OUTPUT_DIR:-approaches/Resnet/results/base_aspp_eagle_vn}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs "$(dirname "$OUTPUT_DIR")"

python3 -m approaches.Resnet.train.train \
    --from_scratch \
    --pretrained_encoder \
    --unfreeze_encoder \
    --use_aspp \
    --use_eam \
    --eam_stages 2 3 \
    --train_dir "$TRAIN_DIR" \
    --gt_depth_dir "$GT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 10 \
    --silog_weight 1.0 \
    --eig_cluster_weight 0.05 \
    --within_cluster_weight 0.1 \
    --vnl_weight 5.0
