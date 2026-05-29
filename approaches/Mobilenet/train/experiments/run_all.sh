#!/bin/bash
# Idempotent dispatcher. Re-run freely — already-complete or already-queued
# experiments are skipped automatically.
#
# Workflow:
#   1) bash approaches/Mobilenet/train/experiments/run_all.sh
#      Submits one job per experiment (5 total).
#   2) If a job hits its time limit, the per-experiment .sh auto-resumes from
#      checkpoints/latest.pth so no state is lost.
#   3) Re-run this script. Done experiments are skipped (history.json has the
#      target epochs); queued/running experiments are skipped (squeue has a
#      matching job name). Only experiments that need another nudge get a fresh
#      sbatch. Repeat until all five are done.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

EXPERIMENTS=(scratch base base_aspp base_aspp_eagle base_aspp_eagle_vn)
TARGET_EPOCHS=20

for exp in "${EXPERIMENTS[@]}"; do
    proj_dir="approaches/Mobilenet/results/${exp}"
    job_name="mobilenet_${exp}"

    # Skip if training is already complete.
    if [[ -f "$proj_dir/history.json" ]]; then
        n=$(python3 -c "
import json
try:
    h = json.load(open('$proj_dir/history.json'))
    print(len(h.get('epochs', [])))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
        if [[ "${n:-0}" -ge "$TARGET_EPOCHS" ]]; then
            echo "skip $exp  -- already complete ($n / $TARGET_EPOCHS epochs)"
            continue
        fi
    fi

    # Skip if a job for this experiment is already queued or running.
    existing=$(squeue -u "$USER" --noheader --name="$job_name" --format="%i" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        echo "skip $exp  -- already in queue (job=$existing)"
        continue
    fi

    jid=$(sbatch --parsable "approaches/Mobilenet/train/experiments/${exp}.sh")
    echo "submit $exp  -> job=$jid"
done

echo
echo "Re-run this script anytime to push experiments forward after each"
echo "time-limit-induced restart. squeue -u \$USER to monitor."
