#!/usr/bin/env bash
# ============================================
# Production pipeline for GW population study
# ============================================

set -euo pipefail
IFS=$'\n\t'

# ---- user configuration ----
SEED=1234
MODEL="powerlaw_peak"
CATALOG="GWTC3"

INPUT_DIR="input_data"
BASE_RUN_DIR="results"

DATA_DIR="${BASE_RUN_DIR}/data"
FIG_DIR="${BASE_RUN_DIR}/figures"
LOG_DIR="${BASE_RUN_DIR}/logs"

# ---- environment ----
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate pop

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true

# ---- directories ----
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${LOG_DIR}"

LOGFILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "${LOGFILE}")
exec 2>&1

echo "=== Starting GW pipeline ==="

# ---- 1. sanity test ----
python sanity_test_methods.py --outdir "${FIG_DIR}" --tag "sanity"

# ---- 2. GWTC-3 inference ----
python run_gwtc3_inference.py --catalog "${CATALOG}" --indir "${INPUT_DIR}" --outdir "${DATA_DIR}/inference" --nsamp-pop 200000 --seed "${SEED}"

# ---- 3. plotting ----
python make_plots.py   --indir "${DATA_DIR}/inference"   --outdir "${FIG_DIR}"

echo "=== Pipeline completed successfully ==="
