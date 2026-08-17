#!/bin/bash
#
# Example GPU submission wrapper for scoring/score_proteins.py. Context defaults
# to the assayed-sequence SQLite DB in agent/data; supply CHECKPOINT.
#
# Cluster settings (account, partition) are intentionally not hardcoded so this
# is portable. Provide them for your scheduler at submit time, and run from the
# agent/ directory, e.g.:
#   sbatch --account=<acct> --partition=<gpu_partition> scoring/run_score.sh
# Override the default checkpoint with CHECKPOINT=/path/to/.../checkpoint.t7.
#
#SBATCH --job-name=pnpt_score
#SBATCH --output=scoring/pnpt_score_%j.out
#SBATCH --error=scoring/pnpt_score_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-model_checkpoints/self_driving/Bgl_spec_2026_round_25/final/checkpoint.t7}"
INPUT_CSV="${INPUT_CSV:-scoring/_test_input.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-scoring/_test_output.csv}"

echo "Host: $(hostname)   GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"

# Activate the project conda environment (see agent/self_driving_env.yml). Assumes
# conda is on PATH -- on module-based clusters you may need e.g. `module load
# anaconda` first. Override the env name with CONDA_ENV, or activate it yourself
# and comment this block out. Never call the env python bare: proteinnpt/torch
# need the env's own libstdc++ ahead of the system one (LD_LIBRARY_PATH below).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-self_driving_env}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Context defaults to the single assayed-sequence DB in agent/data (table
# auto-detected). Override with --database/--table, or bypass with --train_csv.
python scoring/score_proteins.py \
    --input_csv "$INPUT_CSV" \
    --checkpoint "$CHECKPOINT" \
    --output_csv "$OUTPUT_CSV"

echo "=== output ==="
cat "$OUTPUT_CSV"
