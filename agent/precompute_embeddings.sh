source ./config.sh
source activate self_driving_env

export batch_size=1
export max_positions=1024

python precompute_embeddings.py \
    --model_type ${model_type} \
    --model_location ${model_location_sequence_embeddings} \
    --input_data_location ${all_sequences_location} \
    --output_data_location ${embeddings_location} \
    --batch_size ${batch_size} \
    --max_positions ${max_positions} \
    --target_seq ${starting_sequence} \
    --indel_mode \
    --half_precision