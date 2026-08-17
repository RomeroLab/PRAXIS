#!/bin/bash

# Unified dataset_oracle frontier simulation runner.
# Usage: bash insilico_analysis/dataset_oracle/run_frontier.sh PROTEIN DATA_PATH [VALUE_COL] [SEED_SIZE] [WT_FITNESS] [PROPOSAL_MODE] [MAX_POISSON_K] [GPU_INDEX] [SEED] [MODE]
#
# PROTEIN: one of gfp, gb1, pab1, ube4

set -euo pipefail

PROTEIN=${1:?"Error: Provide protein name (gfp, gb1, pab1, ube4) as first argument."}

# Validate protein
case "$PROTEIN" in
    gfp|gb1|pab1|ube4) ;;
    *) echo "Error: Unknown protein '$PROTEIN'. Must be one of: gfp, gb1, pab1, ube4."; exit 1 ;;
esac

# Load protein-specific config for paths and defaults
source ./insilico_analysis/dataset_oracle/${PROTEIN}/config_${PROTEIN}.sh

DATA_PATH=${2:-}
VALUE_COL=${3:-functional_score}
INITIAL_SEED_SIZE=${4:-10}
WT_FITNESS=${5:-}
# Optional: proposal mode (e.g., poisson_neighbors, poisson_sampled_mutants, poisson_exact_n_enumerate)
PROPOSAL_MODE=${6:-}
# Optional: max_poisson_k (e.g., 5)
MAX_POISSON_K=${7:-}
# Optional: GPU index (default 0)
GPU_INDEX=${8:-0}
SEED=${9:-42}
MODE=${10:-}

if [ -z "$DATA_PATH" ]; then
	echo "Error: Provide path to dataset CSV/PKL as second argument."
	echo "Example: bash insilico_analysis/dataset_oracle/run_frontier.sh gb1 /path/to/gb1.csv functional_score 9 1.0 poisson_neighbors 5 0 42 full_pipeline"
	exit 1
fi

# Ensure output dirs
PROTEIN_DIR="insilico_analysis/dataset_oracle/${PROTEIN}"
mkdir -p "$data_location/results"
mkdir -p "$PROTEIN_DIR"
mkdir -p "$zero_shot_fitness_predictions_folder"

# Derive a plain CSV of candidates with 'mutated_sequence' from the input (CSV or PKL)
CAND_DIR="$data_location/data"
mkdir -p "$CAND_DIR"
CAND_CSV="$CAND_DIR/${PROTEIN}_mutants.csv"
python - <<PY
import pandas as pd
path = "$DATA_PATH"
if path.endswith('.pkl') or path.endswith('.pickle'):
    df = pd.read_pickle(path)
elif path.endswith('.tsv') or path.endswith('.txt'):
    df = pd.read_csv(path, sep='\t')
else:
    df = pd.read_csv(path)
col = 'mutated_sequence' if 'mutated_sequence' in df.columns else 'sequence'
df_out = pd.DataFrame({'mutated_sequence': df[col].astype(str).str.replace('*','').str.replace('-','')})
df_out.to_csv("$CAND_CSV", index=False)
print(f"Wrote candidates to $CAND_CSV with {len(df_out)} rows")
PY

# Ensure WT is present in candidates for zero-shot precompute
python - <<PY
import pandas as pd
cand = pd.read_csv("$CAND_CSV")
wt = "$starting_sequence".replace("*","" ).replace("-", "")
if 'mutated_sequence' not in cand.columns:
    cand = cand.rename(columns={cand.columns[0]: 'mutated_sequence'})
if not (cand['mutated_sequence'] == wt).any():
    cand = pd.concat([cand, pd.DataFrame({'mutated_sequence':[wt]})], ignore_index=True)
    cand.to_csv("$CAND_CSV", index=False)
    print("Appended WT to candidates for zero-shot precompute.")
else:
    print("WT already present in candidates.")
PY

# Zero-shot output path (matches precompute_zero_shot.py naming)
ZERO_SHOT_FILE="$zero_shot_fitness_predictions_folder/$(basename "$CAND_CSV" .csv).csv"

# Determine whether existing zero-shot has WT; if not, force re-precompute
NEED_PRECOMPUTE=0
if [ -f "$ZERO_SHOT_FILE" ] && [ -s "$ZERO_SHOT_FILE" ]; then
	set +e
	python - <<PY
import pandas as pd
import sys
wt = "$starting_sequence".replace("*","" ).replace("-", "")
df = pd.read_csv("$ZERO_SHOT_FILE")
if 'mutated_sequence' not in df.columns:
    print("Existing zero-shot file missing 'mutated_sequence' column; will re-precompute.")
    sys.exit(2)
if (df['mutated_sequence'] == wt).any():
    print("Existing zero-shot file includes WT; reuse.")
    sys.exit(0)
else:
    print("Existing zero-shot file missing WT; will re-precompute.")
    sys.exit(2)
PY
	status=$?
	set -e
	if [ $status -ne 0 ]; then
		NEED_PRECOMPUTE=1
	fi
else
	NEED_PRECOMPUTE=1
fi

# Use precomputed zero-shot scores if present and complete; otherwise precompute if checkpoint exists
if [ $NEED_PRECOMPUTE -eq 0 ]; then
  echo "Found precomputed zero-shot at $ZERO_SHOT_FILE. Reusing."
else
  if [ -f "$model_location_zero_shot/config.json" ]; then
    echo "Precomputing zero-shot scores on GPU ${GPU_INDEX}..."
    BATCH=20
    CUDA_VISIBLE_DEVICES=${GPU_INDEX} python precompute_zero_shot.py \
      --checkpoint "$model_location_zero_shot" \
      --batch_size_inference $BATCH \
      --DMS_data_folder "$CAND_DIR" \
      --DMS_file_name "$(basename "$CAND_CSV")" \
      --output_scores_folder "$zero_shot_fitness_predictions_folder" \
      --target_seq "$starting_sequence" \
      --indel_mode \
      --name_score_variable ESM2_650M \
      --scoring_window optimal
  else
    echo "Warning: Tranception checkpoint not found at $model_location_zero_shot. Skipping zero-shot precompute."
    ZERO_SHOT_FILE=""
  fi
fi

# Build args JSON consumed by the simulator
ARGS_JSON="${PROTEIN_DIR}/${PROTEIN}_args.json"
MODE_SUFFIX=""
if [ -n "${MODE}" ]; then
  MODE_SUFFIX="_${MODE}"
fi
MODEL_NAME_SUFFIX=${run_id}_${PROTEIN}_frontier_seed${SEED}${MODE_SUFFIX}
TARGET_CONFIG_JSON="${PROTEIN_DIR}/${PROTEIN}_target_config.json"

cat > "$ARGS_JSON" <<EOF
{
  "data_location": "$data_location",
  "model_config_location": "$model_config_location",
  "target_config_location": "$TARGET_CONFIG_JSON",
  "tablespace_location": "$tablespace_location",
  "database_name": "${database_name%.db}_seed${SEED}${MODE_SUFFIX}.db",
  "table_name": "${table_name}_seed${SEED}${MODE_SUFFIX}",
  "model_checkpoint_folder": "$model_checkpoint_folder",
  "model_name_suffix": "$MODEL_NAME_SUFFIX",
  "starting_sequence": "$starting_sequence",
  "sequence_segments_location": null,
  "target_to_index_mapping": null,
  "sampling_mode": "conditional_sampling",
  "num_acquisitions": $num_acquisitions,
  "acquisition_batch": $acquisition_batch,
  "num_samples_per_iteration": $num_samples_per_iteration,
  "uncertainty_method": "MC_dropout",
  "num_MC_dropout_samples": $num_MC_dropout_samples,
  "selection_method": "top_clusters",
  "acquisition_method": "ucb",
  "ESM_location": "$ESM_location",
  "embedding_model_location": "$ESM_location",
  "MSA_data_folder": null,
  "MSA_weight_data_folder": null,
  "path_to_hhfilter": null,
  "num_data_loaders_workers": 0,
  "max_tokens_per_msa": 16384,
  "early_stopping_patience": 5,
  "max_learning_rate": 0.0003,
  "min_learning_rate": 0.00001,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "label_smoothing": 0.0,
  "grad_norm_clip": 1.0,
  "do_not_save_model_checkpoint": false,
  "fine_tune_model_embedding_parameters": false,
  "use_wandb": false,
  "use_validation_set": false,
  "training_fp16": true,
  "indel_mode": true,
  "num_total_training_steps": $num_total_training_steps
}
EOF

# If zero-shot is available, reference its file only (augmentation remains controlled by model config)
if [ -n "${ZERO_SHOT_FILE:-}" ] && [ -f "$ZERO_SHOT_FILE" ] && [ -s "$ZERO_SHOT_FILE" ]; then
  python - <<PY
import json
p = "$ARGS_JSON"
with open(p,'r') as f:
    o = json.load(f)
o["zero_shot_fitness_predictions_location"] = "$ZERO_SHOT_FILE"
with open(p,'w') as f:
    json.dump(o,f,indent=2)
print("Referenced zero-shot file in args JSON (no augmentation override)")
PY
fi

# Build optional flags
WT_FLAG=""
if [ -n "${WT_FITNESS}" ]; then
  WT_FLAG="--wt_fitness ${WT_FITNESS}"
fi
PROPOSAL_FLAG=""
if [ -n "${PROPOSAL_MODE}" ]; then
  PROPOSAL_FLAG="--proposal_mode ${PROPOSAL_MODE}"
fi
MAXK_FLAG=""
if [ -n "${MAX_POISSON_K}" ]; then
  MAXK_FLAG="--max_poisson_k ${MAX_POISSON_K}"
fi

# Build optional baseline mode flag
MODE_FLAG=""
if [ -n "${MODE}" ] && [ "${MODE}" != "full_pipeline" ]; then
  MODE_FLAG="--baseline_mode ${MODE}"
fi

# Run the unified frontier simulation.
# The simulator seeds the initial labelled set with INITIAL_SEED_SIZE = 10 random
# sequences drawn from the DMS dataset.
OUT_CSV="$data_location/results/${PROTEIN}_frontier_${run_id}_${run_mode}_seed${SEED}${MODE_SUFFIX}.csv"
echo "Running ${PROTEIN} on GPU index ${GPU_INDEX}"
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python insilico_analysis/dataset_oracle/run_frontier_sim.py \
  --protein "$PROTEIN" \
  --args_json "$ARGS_JSON" \
  --data_csv "$DATA_PATH" \
  --value_col "$VALUE_COL" \
  --initial_seed_size "$INITIAL_SEED_SIZE" \
  --seed "$SEED" \
  $WT_FLAG \
  $PROPOSAL_FLAG \
  $MAXK_FLAG \
  $MODE_FLAG \
  --output_csv "$OUT_CSV"

echo "Results written to: $OUT_CSV"
