#!/bin/bash
#
# Master benchmark runner: loops over proteins x methods x seeds (dataset_oracle only).
#
# Usage:
#   bash insilico_analysis/run_full_benchmark.sh [GPU_INDEX] [PROTEIN] [METHOD]
#
# Examples:
#   bash insilico_analysis/run_full_benchmark.sh 0              # all proteins, all methods on GPU 0
#   bash insilico_analysis/run_full_benchmark.sh 0 gfp          # GFP only, all methods on GPU 0
#   bash insilico_analysis/run_full_benchmark.sh 1 gb1          # GB1 only on GPU 1
#   bash insilico_analysis/run_full_benchmark.sh 2 pab1 greedy  # PAB1 greedy only on GPU 2
#
# Parallel across 4 GPUs (one protein per GPU):
#   bash insilico_analysis/run_full_benchmark.sh 0 gfp  &
#   bash insilico_analysis/run_full_benchmark.sh 1 gb1  &
#   bash insilico_analysis/run_full_benchmark.sh 2 pab1 &
#   bash insilico_analysis/run_full_benchmark.sh 3 ube4 &
#   wait
#
# Completed runs are auto-skipped (output CSV exists), so it's safe to re-run.
#
# Datasets default to the per-protein pkl files in insilico_analysis/dataset_oracle/<protein>/.
# Each run is sequential (GPU-bound). Random baseline skips model training (~10x faster).

set -euo pipefail

GPU_INDEX=${1:-0}
FILTER_PROTEIN=${2:-}
FILTER_METHOD=${3:-}
NUM_SEEDS=5
SEEDS=(0 1 2 3 4)
ALL_METHODS=(full_pipeline greedy onehot_mlp onehot_mlp_greedy random)

if [ -n "$FILTER_METHOD" ]; then
    METHODS=("$FILTER_METHOD")
else
    METHODS=("${ALL_METHODS[@]}")
fi

if [ -n "$FILTER_PROTEIN" ]; then
    PROTEINS=("$FILTER_PROTEIN")
else
    PROTEINS=(gfp gb1 pab1 ube4)
fi

BASE_DIR="insilico_analysis/dataset_oracle"

# Protein-specific settings: data_file, value_col, wt_fitness
declare -A PROTEIN_DATA PROTEIN_VALCOL PROTEIN_WT_FITNESS
PROTEIN_DATA[gfp]="${BASE_DIR}/gfp/gfp_SeqFxnDataset.pkl"
PROTEIN_DATA[gb1]="${BASE_DIR}/gb1/gb1_SeqFxnDataset.pkl"
PROTEIN_DATA[pab1]="${BASE_DIR}/pab1/pab1_SeqFxnDataset.pkl"
PROTEIN_DATA[ube4]="${BASE_DIR}/ube4/ube4_SeqFxnDataset.pkl"

PROTEIN_VALCOL[gfp]="functional_score"
PROTEIN_VALCOL[gb1]="functional_score"
PROTEIN_VALCOL[pab1]="functional_score"
PROTEIN_VALCOL[ube4]="functional_score"

PROTEIN_WT_FITNESS[gfp]="3.7"
PROTEIN_WT_FITNESS[gb1]="0.0"
PROTEIN_WT_FITNESS[pab1]="1.0"
PROTEIN_WT_FITNESS[ube4]="1.06"

TOTAL=$((${#SEEDS[@]} * ${#METHODS[@]} * ${#PROTEINS[@]}))
COUNT=0

for PROTEIN in "${PROTEINS[@]}"; do
    DATA_FILE="${PROTEIN_DATA[$PROTEIN]}"
    VALCOL="${PROTEIN_VALCOL[$PROTEIN]}"
    WT_FIT="${PROTEIN_WT_FITNESS[$PROTEIN]}"

    if [ ! -f "$DATA_FILE" ]; then
        echo "WARNING: Dataset not found: $DATA_FILE. Skipping $PROTEIN."
        continue
    fi

    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            echo "=========================================="
            echo "[$COUNT/$TOTAL] ${PROTEIN} / ${METHOD} / seed=${SEED}"
            echo "=========================================="

            # Skip if output CSV already exists (allows resuming after interruption)
            MODE_SUFFIX=""
            if [ -n "$METHOD" ]; then MODE_SUFFIX="_${METHOD}"; fi
            # Source the protein config to get data_location and run_id/run_mode
            source ./insilico_analysis/dataset_oracle/${PROTEIN}/config_${PROTEIN}.sh
            OUT_CSV="$data_location/results/${PROTEIN}_frontier_${run_id}_${run_mode}_seed${SEED}${MODE_SUFFIX}.csv"
            if [ -f "$OUT_CSV" ]; then
                echo "SKIP: Output already exists: $OUT_CSV"
                echo ""
                continue
            fi

            bash insilico_analysis/dataset_oracle/run_frontier.sh \
                "$PROTEIN" "$DATA_FILE" "$VALCOL" 9 "$WT_FIT" "" "" "$GPU_INDEX" "$SEED" "$METHOD"

            echo "Completed: ${PROTEIN} / ${METHOD} / seed=${SEED}"
            echo ""
        done
    done
done

echo "=========================================="
echo "All $COUNT benchmark runs complete."
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Aggregate results per protein/method:"
echo "     python insilico_analysis/aggregate_seeds.py \\"
echo "       --round_log_pattern 'data/insilico/<protein>/results/*_<method>*_round_log.csv' \\"
echo "       --dataset_csv /path/to/<protein>.csv \\"
echo "       --output_dir insilico_analysis/aggregated/<protein>/<method>"
echo ""
echo "  2. Generate paper figures:"
echo "     python insilico_analysis/plot_paper_figures.py \\"
echo "       --results_base_dir insilico_analysis/aggregated \\"
echo "       --output_dir insilico_analysis/figures"
