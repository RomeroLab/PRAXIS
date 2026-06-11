source ./config.sh
source activate self_driving_env

export batch_size_inference=20
all_sequences_folder="${all_sequences_location%/*}"
all_sequences_filename="${all_sequences_location##*/}"

python precompute_zero_shot.py \
        --checkpoint ${model_location_zero_shot} \
        --batch_size_inference ${batch_size_inference} \
        --DMS_data_folder ${all_sequences_folder} \
        --DMS_file_name ${all_sequences_filename} \
        --output_scores_folder ${zero_shot_fitness_predictions_folder} \
        --target_seq ${starting_sequence} \
        --indel_mode \
        --name_score_variable ${model_type} \
        --scoring_window 'optimal'