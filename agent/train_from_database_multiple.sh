#!/bin/bash

# Load shared configuration settings
source ./config.sh
source activate self_driving_env

# Start Agent 1 on GPU 5
CUDA_VISIBLE_DEVICES=5 python train_from_database.py \
        --data_location ${data_location} \
        --model_config_location ${model_config_location} \
        --target_config_location "./configs/targets/self_driving_agent1.json" \
        --target_to_index_mapping ${target_to_index_mapping} \
        --sampling_mode ${sampling_mode} \
        --num_samples_per_iteration ${num_samples_per_iteration} \
        --num_acquisitions ${num_acquisitions} \
        --num_MC_dropout_samples ${num_MC_dropout_samples} \
        --acquisition_batch ${acquisition_batch} \
        --acquisition_method "spec1" \
        --sequence_segments_location ${sequence_segments_location} \
        --starting_sequence ${starting_sequence} \
        --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
        --ESM_location ${ESM_location} \
        --model_checkpoint_folder ${model_checkpoint_folder} \
        --model_name_suffix $run_id"_agent1" \
        --tablespace_location ${tablespace_location} \
        --database_name ${database_name} \
        --table_name ${table_name} \
        --activity_threshold ${activity_threshold} \
        --training_fp16 \
        --indel_mode \
        --num_total_training_steps ${num_total_training_steps} &

# Start Agent 2 on GPU 6
CUDA_VISIBLE_DEVICES=6 python train_from_database.py \
        --data_location ${data_location} \
        --model_config_location ${model_config_location} \
        --target_config_location "./configs/targets/self_driving_agent2.json" \
        --target_to_index_mapping ${target_to_index_mapping} \
        --sampling_mode ${sampling_mode} \
        --num_samples_per_iteration ${num_samples_per_iteration} \
        --num_acquisitions ${num_acquisitions} \
        --num_MC_dropout_samples ${num_MC_dropout_samples} \
        --acquisition_batch ${acquisition_batch} \
        --acquisition_method "spec2" \
        --sequence_segments_location ${sequence_segments_location} \
        --starting_sequence ${starting_sequence} \
        --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
        --ESM_location ${ESM_location} \
        --model_checkpoint_folder ${model_checkpoint_folder} \
        --model_name_suffix $run_id"_agent2" \
        --tablespace_location ${tablespace_location} \
        --database_name ${database_name} \
        --table_name ${table_name} \
        --activity_threshold ${activity_threshold} \
        --training_fp16 \
        --indel_mode \
        --num_total_training_steps ${num_total_training_steps} &

# Start Agent 3 on GPU 7
CUDA_VISIBLE_DEVICES=7 python train_from_database.py \
        --data_location ${data_location} \
        --model_config_location ${model_config_location} \
        --target_config_location "./configs/targets/self_driving_agent3.json" \
        --target_to_index_mapping ${target_to_index_mapping} \
        --sampling_mode ${sampling_mode} \
        --num_samples_per_iteration ${num_samples_per_iteration} \
        --num_acquisitions ${num_acquisitions} \
        --num_MC_dropout_samples ${num_MC_dropout_samples} \
        --acquisition_batch ${acquisition_batch} \
        --acquisition_method "spec3" \
        --sequence_segments_location ${sequence_segments_location} \
        --starting_sequence ${starting_sequence} \
        --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
        --ESM_location ${ESM_location} \
        --model_checkpoint_folder ${model_checkpoint_folder} \
        --model_name_suffix $run_id"_agent3" \
        --tablespace_location ${tablespace_location} \
        --database_name ${database_name} \
        --table_name ${table_name} \
        --activity_threshold ${activity_threshold} \
        --training_fp16 \
        --indel_mode \
        --num_total_training_steps ${num_total_training_steps} & 