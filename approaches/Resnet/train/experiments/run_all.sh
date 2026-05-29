#!/bin/bash
# Run one or all training experiments (the additive ablation ladder).
#
# Usage:
#   bash approaches/Resnet/train/experiments/run_all.sh                  # all 5 configs
#   bash approaches/Resnet/train/experiments/run_all.sh base_aspp_eagle  # just one
#   LOCAL=1 bash approaches/Resnet/train/experiments/run_all.sh          # foreground, no sbatch

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
ALL=(scratch base base_aspp base_aspp_eagle base_aspp_eagle_vn)

# No arg -> all configs. Otherwise run exactly the config given.
if [[ $# -eq 0 ]]; then
    CONFIGS=("${ALL[@]}")
else
    case " ${ALL[*]} " in
        *" $1 "*) CONFIGS=("$1") ;;
        *) echo "unknown config '$1' (choose: ${ALL[*]})"; exit 1 ;;
    esac
fi

for c in "${CONFIGS[@]}"; do
    script="$EXP_DIR/${c}.sh"
    if [[ "${LOCAL:-0}" == "1" ]]; then
        echo "==> running $c locally"
        bash "$script"
    else
        echo "==> submitting $c"
        sbatch "$script"
    fi
done
