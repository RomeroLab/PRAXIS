#!/bin/bash
#
# Round-sweep chimera analysis: score the SAME 1000 chimeras with every round's
# model checkpoint against that round's truncated training database, producing one
# predictions CSV per round.
#
#   round r  ->  checkpoint round_r  +  DB sliced to round r's created_at cutoff
#               (built by scoring/build_round_dbs.py)  ->  predictions_round_r.csv
#
# The chimera set is GENERATED at runtime from a fixed seed (not read from a
# checked-in file), exactly once, and shared by every round -- so the run is
# reproducible from scratch and predictions are directly comparable across rounds.
# generate_chimeras.py is deterministic given (--seed, segments, exclusion DB);
# SEED=0 reproduces the canonical set. Run scoring/build_round_dbs.py once first to
# create scoring/round_dbs/.
#
# Cluster settings (account, partition) are intentionally not hardcoded. Provide
# them at submit time and run from the agent/ directory, e.g.:
#   sbatch --account=<acct> --partition=<gpu_partition> scoring/run_round_sweep.sh
#
#SBATCH --job-name=pnpt_round_sweep
#SBATCH --output=scoring/round_sweep/sweep_%A_%a.out
#SBATCH --error=scoring/round_sweep/sweep_%A_%a.err
#SBATCH --array=0-24%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail

R="${SLURM_ARRAY_TASK_ID:?run as a SLURM array job, e.g. --array=0-24}"

N="${N:-1000}"
SEED="${SEED:-0}"
CKPT_ROOT="${CKPT_ROOT:-data/round_checkpoints}"
DB_DIR="${DB_DIR:-scoring/round_dbs}"
OUT_DIR="${OUT_DIR:-scoring/round_sweep}"

CHECKPOINT="$CKPT_ROOT/Bgl_spec_2026_round_${R}/final/checkpoint.t7"
DATABASE="$DB_DIR/Bgl_spec_2026_round_${R}.db"
OUTPUT="$OUT_DIR/predictions_round_${R}.csv"
# One shared, runtime-generated chimera set (in the output dir, not the repo).
CHIMERAS="$OUT_DIR/chimeras_seed${SEED}_n${N}.csv"

mkdir -p "$OUT_DIR"

echo "Host: $(hostname)   GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ROUND=$R  N=$N  SEED=$SEED"
echo "  checkpoint: $CHECKPOINT"
echo "  database  : $DATABASE"
echo "  chimeras  : $CHIMERAS"
echo "  output    : $OUTPUT"

[ -f "$CHECKPOINT" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[ -f "$DATABASE" ]   || { echo "ERROR: round DB not found (run build_round_dbs.py first): $DATABASE" >&2; exit 1; }

# Use the project conda env (see agent/self_driving_env.yml) directly by path --
# some compute nodes have no conda on PATH and no conda.sh, so `conda activate`
# is unavailable. Putting the env's bin first makes `python` the env interpreter,
# and its lib first satisfies proteinnpt/torch's need for the env's own libstdc++
# ahead of the system one (the sole reason activation mattered). Set
# CONDA_ENV_PATH to the env prefix if the default below does not resolve.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$(conda info --base 2>/dev/null)/envs/${CONDA_ENV:-self_driving_env}}"
[ -x "$CONDA_ENV_PATH/bin/python" ] || { echo "ERROR: env python not found at $CONDA_ENV_PATH (set CONDA_ENV_PATH)" >&2; exit 1; }
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"

# 1. Generate the fixed chimera set ONCE, shared across the array. Deterministic
#    (seed + exclusion DB fixed), so whichever task wins the lock, every round
#    scores the identical 1000 chimeras. Lock via mkdir (atomic); others wait.
if [ ! -f "$CHIMERAS" ]; then
    if mkdir "$CHIMERAS.lock" 2>/dev/null; then
        echo "Task $R generating chimeras -> $CHIMERAS"
        python scoring/generate_chimeras.py --n "$N" --seed "$SEED" --output "$CHIMERAS.tmp"
        mv "$CHIMERAS.tmp" "$CHIMERAS"
        rmdir "$CHIMERAS.lock"
    else
        echo "Task $R waiting for another task to generate $CHIMERAS ..."
        for _ in $(seq 1 120); do [ -f "$CHIMERAS" ] && break; sleep 5; done
        [ -f "$CHIMERAS" ] || { echo "ERROR: chimeras not produced within timeout" >&2; exit 1; }
    fi
fi

# 2. Score the shared chimera set against this round's model + truncated DB.
#    EVAL_BATCH (optional) caps the per-batch chimera count to fit large-context
#    rounds in GPU memory.
python scoring/score_proteins.py \
    --input_csv "$CHIMERAS" \
    --checkpoint "$CHECKPOINT" \
    --database "$DATABASE" \
    --output_csv "$OUTPUT" \
    ${EVAL_BATCH:+--eval_batch_size "$EVAL_BATCH"}

echo "=== round $R done -> $OUTPUT ==="
head -5 "$OUTPUT"
