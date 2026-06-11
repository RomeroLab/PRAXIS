import h5py
import torch
import argparse
import glob
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, 
                       default='data/embeddings/self_driving/ESM2_650M',
                       help='Directory containing GPU output files')
    parser.add_argument('--output_file', type=str, 
                       default='data/embeddings/self_driving/ESM2_650M/GH1_all_sequences.h5',
                       help='Path to final combined file')
    return parser.parse_args()

def check_files(input_files):
    """Verify files exist and are valid h5 files"""
    valid_files = []
    for file in input_files:
        if os.path.isfile(file):
            try:
                with h5py.File(file, 'r') as f:
                    if 'embeddings' in f:
                        valid_files.append(file)
                    else:
                        print(f"Warning: {file} does not contain embeddings dataset")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"Warning: {file} is not a file or doesn't exist")
    return valid_files

def get_max_length_from_files(input_files):
    max_len = 0
    for file in input_files:
        try:
            with h5py.File(file, 'r') as f:
                if 'embeddings' in f:
                    curr_len = f['embeddings'].shape[1]
                    max_len = max(max_len, curr_len)
                    print(f"File {file}: sequence length = {curr_len}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return max_len

def combine_h5_files(input_dir, output_file):
    # Get list of input files
    input_files = []
    for gpu in range(4, 8):  # GPUs 4-7
        gpu_dir = os.path.join(input_dir, f'GH1_all_sequences_gpu{gpu}.h5')
        h5_file = os.path.join(gpu_dir, 'GH1_all_sequences.h5')
        if os.path.exists(h5_file):
            input_files.append(h5_file)
    
    if not input_files:
        raise ValueError(f"No GPU output files found in {input_dir}")
    
    print(f"Found {len(input_files)} files to combine:")
    for f in input_files:
        print(f"  {f}")
    
    # Verify files are valid
    valid_files = check_files(input_files)
    if not valid_files:
        raise ValueError("No valid HDF5 files found")
    print(f"Found {len(valid_files)} valid files to combine")
    
    # Get global maximum sequence length
    max_seq_len = get_max_length_from_files(valid_files)
    print(f"Maximum sequence length across all files: {max_seq_len}")
    
    # Open output file
    with h5py.File(output_file, 'w') as out_f:
        # Process each input file
        total_seqs = 0
        for i, in_file in enumerate(valid_files):
            print(f"Processing file {i+1}/{len(valid_files)}: {in_file}")
            try:
                with h5py.File(in_file, 'r') as in_f:
                    # For first file, create datasets
                    if i == 0:
                        for key in in_f.keys():
                            shape = list(in_f[key].shape)
                            if key == 'embeddings':
                                shape[1] = max_seq_len
                            maxshape = list(shape)
                            maxshape[0] = None
                            out_f.create_dataset(key, shape=shape, maxshape=maxshape, 
                                              dtype=in_f[key].dtype)
                    
                    # For all files, append data
                    for key in in_f.keys():
                        in_data = in_f[key][:]
                        out_dset = out_f[key]
                        old_size = out_dset.shape[0]
                        new_size = old_size + in_data.shape[0]
                        out_dset.resize(new_size, axis=0)
                        
                        if key == 'embeddings' and in_data.shape[1] != max_seq_len:
                            pad_len = max_seq_len - in_data.shape[1]
                            padded_data = torch.zeros(in_data.shape[0], max_seq_len, in_data.shape[2], 
                                                    dtype=torch.float16 if in_data.dtype == torch.float16 else torch.float32)
                            padded_data[:, :in_data.shape[1], :] = torch.tensor(in_data)
                            out_dset[old_size:new_size] = padded_data
                        else:
                            out_dset[old_size:new_size] = in_data
                    
                    total_seqs += in_data.shape[0]
                    print(f"Processed {in_data.shape[0]} sequences, total: {total_seqs}")
            except Exception as e:
                print(f"Error processing file {in_file}: {e}")

if __name__ == '__main__':
    args = parse_arguments()
    print(f"Combining files from {args.input_dir}")
    print(f"Output will be saved to {args.output_file}")
    combine_h5_files(args.input_dir, args.output_file)
    print("Combination complete!")