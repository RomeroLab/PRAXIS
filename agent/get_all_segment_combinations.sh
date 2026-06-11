source ./config.sh
source activate self_driving_env

python get_all_segment_combinations.py \
        --sequence_segments_location ${sequence_segments_location} \
        --all_sequences_location ${all_sequences_location}    