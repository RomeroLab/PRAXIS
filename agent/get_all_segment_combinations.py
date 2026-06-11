import os
import pandas as pd
import itertools
from utils import convert_fragment_df_to_dict
import argparse

def get_all_combinations(sequence_segments):
    """Simple function to get all possible combinations"""
    segment_keys = sorted(sequence_segments)
    segments_in_order = [sequence_segments[key] for key in segment_keys]
    all_combinations = itertools.product(*segments_in_order)
    return [''.join(combination) for combination in all_combinations]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument('--sequence_segments_location', type=str, help='Path to csv file that contains sequence segments')
    parser.add_argument('--all_sequences_location', type=str, help='Path to location where we store a csv w/ all possible sequence segments combinations')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Get all possible sequences
    if not os.path.exists(args.all_sequences_location):
        sequence_segments = convert_fragment_df_to_dict(pd.read_csv(args.sequence_segments_location, index_col=0, low_memory=False))
        all_sequences = get_all_combinations(sequence_segments)
        df = pd.DataFrame(all_sequences, columns=['mutated_sequence'])
        parent_directory = os.path.dirname(args.all_sequences_location)
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
        df.to_csv(args.all_sequences_location, index=False)
    else:
        print("### File with all possible segment combinations already exists at specified location ###")