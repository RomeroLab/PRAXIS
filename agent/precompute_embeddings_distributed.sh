#!/bin/bash
source ./config.sh
source activate self_driving_env

# Calculate sequences per GPU
total_seqs=$(wc -l < ${all_sequences_location})
seqs_per_gpu=$(( (total_seqs + 3) / 4 ))

# Launch jobs on GPUs 4-7 instead of 0-3
for gpu_id in {4..7}; do
    # Calculate chunk range (still using 0-3 for indexing chunks)
    chunk_id=$((gpu_id - 4))  # Convert GPU ID 4-7 to chunk index 0-3
    start_idx=$((chunk_id * seqs_per_gpu))
    end_idx=$(( (chunk_id + 1) * seqs_per_gpu ))
    
    # Ensure we don't exceed total sequences
    if [ $end_idx -gt $total_seqs ]; then
        end_idx=$total_seqs
    fi
    
    echo "Starting GPU $gpu_id with sequences $start_idx to $end_idx"
    
    # Set output file for this GPU
    gpu_output="${embeddings_location%.*}_gpu${gpu_id}.h5"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python precompute_embeddings.py \
        --model_type ${model_type} \
        --model_location ${model_location_sequence_embeddings} \
        --input_data_location ${all_sequences_location} \
        --output_data_location ${gpu_output} \
        --batch_size 1 \
        --max_positions 1024 \
        --long_sequences_slicing_method "center" \
        --target_seq ${starting_sequence} \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --indel_mode \
        --half_precision &
    
    echo "Launched job on GPU $gpu_id"
done

# Wait for all GPU jobs to complete
wait

echo "All GPU jobs completed. Combining results..."

# Combine results
python combine_embeddings.py \
    --input_pattern "${embeddings_location%.*}_gpu*.h5" \
    --output_file ${embeddings_location}

# Clean up intermediate files
rm ${embeddings_location%.*}_gpu*.h5

echo "Process complete!"