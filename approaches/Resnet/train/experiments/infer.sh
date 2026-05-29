#!/bin/bash
#SBATCH --job-name=resnet_infer
#SBATCH --time=04:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=studgpu-node09
#SBATCH --output=logs/resnet_infer_%j.out
#SBATCH --error=logs/resnet_infer_%j.err
#
# Inference + submission for a trained run.
# Point RUN_DIR at a run folder that contains best_encoder/ and
# best_depth_head.pt (default: the latest base_aspp_eagle_vn run).
# Pass USE_ASPP=0 if the run was trained WITHOUT --use_aspp.

set -euo pipefail

module load cuda/12.8.0 || true
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

TEST_DIR="${TEST_DIR:-/cluster/courses/cil/monocular-depth-estimation/test}"
RUN_DIR="${RUN_DIR:-$(ls -d approaches/Resnet/results/base_aspp_eagle_vn/*/ 2>/dev/null | head -n 1)}"
OUTPUT_DIR="${OUTPUT_DIR:-approaches/Resnet/results/predictions}"
OUTPUT_CSV="${OUTPUT_CSV:-submission.csv}"

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "No RUN_DIR found. Set RUN_DIR=<path to run with best_encoder/>." >&2
    exit 1
fi
echo "Using checkpoints from: ${RUN_DIR}"

ASPP_FLAG="--use_aspp"
[[ "${USE_ASPP:-1}" == "0" ]] && ASPP_FLAG=""

python3 -m approaches.Resnet.inference.inference \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --encoder_dir "${RUN_DIR}best_encoder" \
    --head_path "${RUN_DIR}best_depth_head.pt" \
    --input_size 256 \
    $ASPP_FLAG
