#!/bin/bash
#SBATCH --job-name=mobilenet_base_aspp_eagle_vn
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/mobilenet_base_aspp_eagle_vn_%j.out
#SBATCH --error=logs/mobilenet_base_aspp_eagle_vn_%j.err
#
# Experiment 5 (top-line): + Virtual Normal geometric loss
#   - everything in base_aspp_eagle.sh
#   - + Virtual Normal (Yin et al. ICCV 2019), weight 5.0, n_triplets 1024
# Isolates the contribution of the geometric VN prior on top of base_aspp_eagle.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

TRAIN_DIR="${TRAIN_DIR:-/cluster/courses/cil/monocular-depth-estimation/train}"
TEST_DIR="${TEST_DIR:-/cluster/courses/cil/monocular-depth-estimation/test}"
GT_DIR="${GT_DIR:-${CIL_CLEANED_GT_DIR:-./train_depth_clean}}"
PROJ_DIR="${PROJ_DIR:-approaches/Mobilenet/results/base_aspp_eagle_vn}"

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
    --proj_dir     "$PROJ_DIR" \
    --train_dir    "$TRAIN_DIR" \
    --test_dir     "$TEST_DIR" \
    --gt_depth_dir "$GT_DIR" \
    --image_size 576 \
    --epochs 20 \
    --batch_size 8 \
    --num_workers 8 \
    --lr 3e-4 \
    --warmup_steps 700 \
    --pretrained_encoder \
    --use_aspp \
    --eam_scales 32 16 \
    --eam_k 4 \
    --num_clusters 10 \
    --eam_sigma_color 0.2 \
    --silog_weight 1.0 \
    --eig_cluster_weight 0.05 \
    --within_cluster_weight 0.1 \
    --virtual_normal_weight 5.0 \
    --virtual_normal_n_triplets 1024 \
    --virtual_normal_fov_deg 60.0
