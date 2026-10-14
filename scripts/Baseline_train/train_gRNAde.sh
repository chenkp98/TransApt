#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="${BASE_DIR}/geometric-rna-design-main"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate rna

export LD_PRELOAD="${CONDA_PREFIX}/../bio_tools/lib/libstdc++.so.6"

export PROJECT_PATH="${PROJECT_PATH}"
export DATA_PATH="${PROJECT_PATH}/data"
export ETERNAFOLD="${PROJECT_PATH}"
export X3DNA="${PROJECT_PATH}"

export WANDB_PROJECT="grnade-custom-training"
export WANDB_ENTITY="chenh"

LOG_FILE="train_grnade_$(date +'%Y%m%d_%H%M%S').log"
CONFIG_PATH="${PROJECT_PATH}/configs/design.yaml"

echo "=== Starting gRNAde (Logs: ${LOG_FILE}) ==="

python3 "${PROJECT_PATH}/design.py" \
    --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"