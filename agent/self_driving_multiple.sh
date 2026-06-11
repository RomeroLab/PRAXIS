#!/bin/bash

# Load shared configuration settings
source ./config.sh
source activate self_driving_env

# Single-GPU multi-agent run: one shared model, three acquisition strategies
CUDA_VISIBLE_DEVICES=2 python self_driving.py \
        --data_location ${data_location} \
        --model_config_location ${model_config_location} \
        --target_config_location "./configs/targets/self_driving_agent1.json" \
        --target_to_index_mapping ${target_to_index_mapping} \
        --sampling_mode ${sampling_mode} \
        --num_samples_per_iteration ${num_samples_per_iteration} \
        --num_acquisitions ${num_acquisitions} \
        --num_MC_dropout_samples ${num_MC_dropout_samples} \
        --acquisition_batch ${acquisition_batch} \
        --acquisition_methods "spec1,spec2,spec3" \
        --sequence_segments_location ${sequence_segments_location} \
        --starting_sequence ${starting_sequence} \
        --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
        --ESM_location ${ESM_location} \
        --model_checkpoint_folder ${model_checkpoint_folder} \
        --model_name_suffix ${run_id} \
        --tablespace_location ${tablespace_location} \
        --database_name ${database_name} \
        --table_name ${table_name} \
        --activity_threshold ${activity_threshold} \
        --training_fp16 \
        --indel_mode \
        --num_total_training_steps ${num_total_training_steps} \
        ${seed_sequences:+--seed_sequences "$seed_sequences"} \
        ${starting_round:+--starting_round $starting_round}
