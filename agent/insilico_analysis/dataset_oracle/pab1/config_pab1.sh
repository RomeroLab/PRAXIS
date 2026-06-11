#!/bin/bash

# GFP frontier simulation config (final mode)
# Shared variables reused from the main project and GFP-specific overrides

# Mode and run identification
export run_mode="final"
export run_id="pab1"

# --- Resolve repository root (the agent/ directory) if not already provided ---
if [ -z "${PRAXIS_ROOT:-}" ]; then
  _praxis_d="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  while [ "$_praxis_d" != "/" ] && [ ! -f "$_praxis_d/self_driving.py" ]; do
    _praxis_d="$( dirname "$_praxis_d" )"
  done
  export PRAXIS_ROOT="$_praxis_d"
fi

# Paths
export data_location=${PRAXIS_ROOT}/insilico_analysis/dataset_oracle/pab1
export model_config_location="./configs/model/PNPT_ESM2_650M_final.json"  # shared model config
export model_checkpoint_folder=$data_location"/model_checkpoints/pab1_synevo"
export tablespace_location=$data_location"/data/tablespace/pab1_synevo"
export database_name=${run_id}"_"${run_mode}"_database.db"
export table_name=${run_id}"_"${run_mode}"_assayed_sequences"
# Absolute path to ESM2 (650M) checkpoint under self_driving/ESM/ESM2
export ESM_location=${PRAXIS_ROOT}/ESM/ESM2/esm2_t33_650M_UR50D.pt
# Tranception zero-shot checkpoint (reads only)
export model_location_zero_shot=${PRAXIS_ROOT}/Tranception/Tranception_Large/Tranception_Large
# Local folder to store zero-shot predictions
export zero_shot_fitness_predictions_folder=$data_location"/data/zero_shot_fitness_predictions"

# pab1 WT (ensure this matches the dataset reference sequence)
export starting_sequence="GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP"

# Training/AL settings (final-like defaults)
export num_total_training_steps=5000
export num_acquisitions=20
export num_MC_dropout_samples=5
export acquisition_batch=10
export num_samples_per_iteration=1000 