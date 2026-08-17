#!/bin/bash
#
# Chimera predictive-capacity analysis:
#   1. generate N random chimeras by stitching sequence segments
#   2. score them with a ProteinNPT checkpoint + the assayed-sequence DB context
#   3. write predictions.csv (chimera, sequence, predicted a1/a2/a3)
#
# Cluster settings (account, partition) are intentionally not hardcoded so this
# is portable. Provide them for your scheduler at submit time, and run from the
# agent/ directory, e.g.:
#   sbatch --account=<acct> --partition=<gpu_partition> scoring/run_chimera_analysis.sh
#   N=1000 SEED=0 CHECKPOINT=... OUTPUT=scoring/predictions.csv \
#     sbatch --account=<acct> --partition=<gpu_partition> scoring/run_chimera_analysis.sh
#
#SBATCH --job-name=pnpt_chimera
#SBATCH --output=scoring/pnpt_chimera_%j.out
#SBATCH --error=scoring/pnpt_chimera_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

N="${N:-1000}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-model_checkpoints/self_driving/Bgl_spec_2026_round_25/final/checkpoint.t7}"
CHIMERAS="${CHIMERAS:-scoring/chimeras.csv}"
OUTPUT="${OUTPUT:-scoring/predictions.csv}"

echo "Host: $(hostname)   GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "N=$N SEED=$SEED CHECKPOINT=$CHECKPOINT OUTPUT=$OUTPUT"

# Activate the project conda environment (see agent/self_driving_env.yml). Assumes
# conda is on PATH -- on module-based clusters you may need e.g. `module load
# anaconda` first. Override the env name with CONDA_ENV. Never call the env python
# bare: proteinnpt/torch need the env's own libstdc++ ahead of the system one.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-self_driving_env}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 1. Generate chimeras (excludes sequences already in the context DB).
python scoring/generate_chimeras.py \
    --n "$N" --seed "$SEED" --output "$CHIMERAS"

# 2. Score them against the DB context.
python scoring/score_proteins.py \
    --input_csv "$CHIMERAS" \
    --checkpoint "$CHECKPOINT" \
    --output_csv "$OUTPUT"

echo "=== predictions.csv head ==="
head -10 "$OUTPUT"
echo "=== summary ==="
python - "$OUTPUT" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
print("rows:", len(d))
for c in ("a1", "a2", "a3"):
    if c in d:
        s = d[c]
        print(f"  {c}: mean={s.mean():.3f} std={s.std():.3f} min={s.min():.3f} max={s.max():.3f}")
PY
