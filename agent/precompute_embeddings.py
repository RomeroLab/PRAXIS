import os,sys
import torch
import numpy as np
import pandas as pd
import argparse
import h5py
import json
import tqdm
import gc
import psutil

from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.tranception.config import TranceptionConfig
from proteinnpt.utils.tranception.model_pytorch import TranceptionLMHeadModel
from proteinnpt.utils.esm.pretrained import load_model_and_alphabet
from proteinnpt.utils.msa_utils import process_MSA
from proteinnpt.utils.data_utils import cleanup_ids_assay_data
from utils import *

def check_memory_usage(safety_threshold=0.90):
    """
    Check if memory usage is approaching dangerous levels
    Returns True if safe, False if memory usage too high
    """
    memory = psutil.virtual_memory()
    memory_usage = memory.percent / 100.0
    
    if memory_usage > safety_threshold:
        print(f"WARNING: Memory usage at {memory_usage*100:.1f}% exceeds safety threshold of {safety_threshold*100:.1f}%")
        return False
    return True

def get_max_sequence_length(input_data_location, indel_mode=False, target_seq=None):
    """Get maximum sequence length from all data including processing steps"""
    max_len = 0
    for chunk in pd.read_csv(input_data_location, chunksize=10000):
        if 'mutated_sequence' in chunk.columns:
            # Apply same preprocessing as in main processing
            sequences = chunk['mutated_sequence'].apply(lambda x: x.replace("*",""))
            sequences = sequences.apply(lambda x: x.replace("-",""))  # Remove gaps
            if target_seq is not None:
                # Apply any cleanup from cleanup_ids_assay_data if needed
                chunk = cleanup_ids_assay_data(chunk, indel_mode=indel_mode, target_seq=target_seq)
                sequences = chunk['mutated_sequence']
            chunk_max = sequences.str.len().max()
        elif 'sequence' in chunk.columns:
            sequences = chunk['sequence'].apply(lambda x: x.replace("*",""))
            sequences = sequences.apply(lambda x: x.replace("-",""))
            chunk_max = sequences.str.len().max()
        max_len = max(max_len, chunk_max)
    
    # Add any necessary padding tokens
    if args.model_type.startswith("ESM"):
        max_len += 2  # For BOS and EOS tokens
    
    return max_len

def load_sequences_chunk(input_data_location, start_idx, end_idx):
    """Load sequences efficiently using chunks"""
    # First check columns in file
    header_df = pd.read_csv(input_data_location, nrows=0)
    available_columns = header_df.columns.tolist()
    
    # Define efficient dtypes for the columns we need
    dtype_dict = {}
    
    # Check which columns exist and use appropriate names
    cols_to_use = []
    if 'mutated_sequence' in available_columns:
        cols_to_use.append('mutated_sequence')
        dtype_dict['mutated_sequence'] = 'string'
    elif 'sequence' in available_columns:
        cols_to_use.append('sequence')
        dtype_dict['sequence'] = 'string'
        
    if 'mutant' in available_columns:
        cols_to_use.append('mutant')
        dtype_dict['mutant'] = 'string'
    elif 'mutation' in available_columns:
        cols_to_use.append('mutation')
        dtype_dict['mutation'] = 'string'
        
    if not cols_to_use:
        raise ValueError(f"Could not find required columns in file. Available columns: {available_columns}")
    
    # Read only the required rows and columns
    df_chunk = pd.read_csv(input_data_location, 
                          skiprows=range(1, start_idx + 1) if start_idx > 0 else None,
                          nrows=end_idx - start_idx if end_idx else None,
                          usecols=cols_to_use,
                          dtype=dtype_dict)
    
    # Rename columns if needed to match expected names
    column_mapping = {
        'sequence': 'mutated_sequence',
        'mutation': 'mutant'
    }
    df_chunk = df_chunk.rename(columns=column_mapping)
    
    return df_chunk

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument('--assay_reference_file_location', default=None, type=str, help='Path to reference file with list of assays to score')
    parser.add_argument('--assay_index', default=0, type=int, help='Index of assay in the ProteinGym reference file to compute embeddings for')
    parser.add_argument('--input_data_location', default=None, type=str, help='Location of input data with all mutated sequences')
    parser.add_argument('--output_data_location', default=None, type=str, help='Location of output embeddings')
    parser.add_argument('--model_type', default='Tranception', type=str, help='Model type to compute embeddings with ["MSA_Transformer", "ESM1v", "Tranception"]')
    parser.add_argument('--model_location', default=None, type=str, help='Location of model used to embed protein sequences')
    parser.add_argument('--max_positions', default=1024, type=int, help='Maximum context length of embedding model')
    parser.add_argument('--long_sequences_slicing_method', default='center', type=str, help='Method to slice long sequences [rolling, center, left]')
    parser.add_argument('--batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--indel_mode', action='store_true', help='Use this mode if extracting embeddings of indel assays')
    parser.add_argument('--half_precision', action='store_true', help='Store embeddings as 16-bit floating point numbers (float16)')
    parser.add_argument('--num_MSA_sequences', default=1000, type=int, help='Num MSA sequences to score each sequence with')
    parser.add_argument('--MSA_data_folder', default=None, type=str, help='Folder where all MSAs are stored')
    parser.add_argument('--MSA_weight_data_folder', default=None, type=str, help='Folder where MSA sequence weights are stored')
    parser.add_argument('--path_to_hhfilter', default=None, type=str, help='Path to hhfilter (for filtering MSAs)')
    parser.add_argument('--path_to_clustalomega', default=None, type=str, help='Path to clustal omega')
    parser.add_argument('--fast_MSA_mode', action='store_true', help='Use this mode to speed up embedding extraction for MSA Transformer')
    parser.add_argument('--target_seq', default=None, type=str, help='Wild type sequence mutated in the assay')
    parser.add_argument('--MSA_location', default=None, type=str, help='Path to MSA file (.a2m)')
    parser.add_argument('--weight_file_name', default=None, type=str, help='Name of weight file')
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA')
    parser.add_argument("--use_cpu", action="store_true", help="Force the use of CPU instead of GPU")
    parser.add_argument('--start_idx', type=int, default=None, help='Start index of sequences to process')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of sequences to process')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Number of sequences to process at once')
    return parser.parse_args()

def preprocess_gaps_in_sequences(df, mode="drop_gaps", MSA_sequences=None):
    if mode=="drop_gaps":
        df['mutated_sequence'] = df['mutated_sequence'].apply(lambda x: x.replace("-",""))
        return df, MSA_sequences
    elif mode=="insert_focus_seq_gaps_in_MSA":
        first_sequence = df['mutated_sequence'].values[0]
        dash_positions = [i for i, char in enumerate(first_sequence) if char == "-"]
        if len(dash_positions)>0:
            print("Gaps detected in reference sequence -- inserting gaps in the MSA accordingly")
            MSA_sequences_with_dashes = []
            for seq_id, sequence in MSA_sequences:
                sequence_list = list(sequence)
                for position in dash_positions:
                    sequence_list.insert(position, "-")
                sequence_with_dashes = ''.join(sequence_list)
                MSA_sequences_with_dashes.append((seq_id, sequence_with_dashes))
            MSA_sequences = MSA_sequences_with_dashes
        return df, MSA_sequences

def save_chunk_results(output_path, chunk_idx, embeddings_dict):
    """Save chunk results to H5 file"""
    with h5py.File(output_path, 'a') as h5f:
        for key, value in embeddings_dict.items():
            if chunk_idx == 0 and key not in h5f:
                h5f.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:] if hasattr(value, 'shape') else (None,))
            else:
                dset = h5f[key]
                current_len = dset.shape[0]
                new_len = current_len + (len(value) if hasattr(value, '__len__') else 1)
                dset.resize(new_len, axis=0)
                dset[current_len:] = value

def main(args):
    if not args.use_cpu and not torch.cuda.is_available():
        print("Error: CUDA not available. This script is intended to run on a GPU, to use a CPU run with --use_cpu")
        exit(1)
    elif args.use_cpu and torch.cuda.is_available():
        print("Note: CUDA is available, but will not be used because --use_cpu is specified.")

    try:
        # Set up paths and load model
        MSA_filename = None
        if args.assay_reference_file_location is not None:
            assay_reference_file = pd.read_csv(args.assay_reference_file_location)
            assay_id = assay_reference_file["DMS_id"][args.assay_index]
            assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0]
            args.input_data_location = args.input_data_location + os.sep + assay_file_name
            args.target_seq = assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==assay_id].values[0]
        else:
            assert args.target_seq is not None, "Reference file provided and target_seq not provided"
            assay_id = args.input_data_location.split(".csv")[0].split(os.sep)[-1]
            assay_file_name = args.input_data_location.split(os.sep)[-1]
            if args.MSA_location:
                MSA_filename = args.MSA_location.split(os.sep)[-1]
                args.MSA_data_folder = os.sep.join(args.MSA_location.split(os.sep)[:-1])
            if (args.MSA_start is None) or (args.MSA_end is None): 
                if args.MSA_data_folder: print("MSA start or MSA end not provided -- Assuming the MSA is covering the full WT sequence")
                args.MSA_start = 1
                args.MSA_end = len(args.target_seq)

        print("Assay: {}".format(assay_file_name))

        global_max_length = get_max_sequence_length(
            args.input_data_location, 
            indel_mode=args.indel_mode,
            target_seq=args.target_seq
        )
        print(f"Maximum sequence length across all data: {global_max_length}")

        # Load model
        if args.model_type=="MSA_Transformer" or args.model_type.startswith("ESM"):
            model, alphabet = load_model_and_alphabet(args.model_location)
            alphabet_size = len(alphabet)
            model.MSA_sample_sequences=None
        elif args.model_type=="Tranception":
            config = json.load(open(args.model_location+os.sep+'config.json'))
            config = TranceptionConfig(**config)
            config.tokenizer = get_tranception_tokenizer()
            config.full_target_seq = args.target_seq
            config.inference_time_retrieval_type = None
            config.retrieval_aggregation_mode = None
            alphabet = None
            model = TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.model_location,config=config)

        # Setup model
        model.eval()
        if not args.use_cpu: 
            model.cuda()

        # Calculate total sequences and chunks
        total_lines = sum(1 for _ in open(args.input_data_location)) - 1
        if args.end_idx is None:
            args.end_idx = total_lines
        if args.start_idx is None:
            args.start_idx = 0

        # Create output path
        if not os.path.exists(args.output_data_location):
            os.makedirs(args.output_data_location)
        output_embeddings_path = args.output_data_location + os.sep + assay_file_name.split(".csv")[0] + '.h5'


        # Process chunks
        num_chunks = (args.end_idx - args.start_idx + args.chunk_size - 1) // args.chunk_size
        
        for chunk_idx in range(num_chunks):
            if not check_memory_usage():
                print("Emergency save: Memory usage too high")
                break
                
            chunk_start = args.start_idx + (chunk_idx * args.chunk_size)
            chunk_end = min(args.start_idx + ((chunk_idx + 1) * args.chunk_size), args.end_idx)
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (sequences {chunk_start} to {chunk_end})")
            
            try:
                # Load and process chunk
                df_chunk = load_sequences_chunk(args.input_data_location, chunk_start, chunk_end)
                df_chunk['mutated_sequence'] = df_chunk['mutated_sequence'].apply(lambda x: x.replace("*",""))
                df_chunk = cleanup_ids_assay_data(df_chunk, indel_mode=args.indel_mode, target_seq=args.target_seq)
                df_chunk = df_chunk[['mutant','mutated_sequence']]

                # Process MSA if needed
                if args.model_type=="MSA_Transformer":
                    MSA_sequences, MSA_weights = process_MSA(
                        MSA_data_folder=args.MSA_data_folder,
                        MSA_weight_data_folder=args.MSA_weight_data_folder,
                        MSA_filename=MSA_filename,
                        MSA_weights_filename=args.weight_file_name,
                        path_to_hhfilter=args.path_to_hhfilter
                    )
                else:
                    MSA_sequences, MSA_weights = None, None

                # Preprocess gaps
                if args.model_type=="Tranception" or args.model_type.startswith("ESM"):
                    df_chunk, MSA_sequences = preprocess_gaps_in_sequences(df_chunk, mode="drop_gaps")
                elif args.model_type=="MSA_Transformer":
                    df_chunk, MSA_sequences = preprocess_gaps_in_sequences(
                        df_chunk,
                        mode="insert_focus_seq_gaps_in_MSA",
                        MSA_sequences=MSA_sequences
                    )

                # Create dataloader
                if args.model_type=="MSA_Transformer" or args.model_type.startswith("ESM"):
                    dataloader = get_ESM_dataloader(df_chunk, args.batch_size)
                elif args.model_type=="Tranception":
                    dataloader = get_Tranception_dataloader(df_chunk, args.batch_size, model, args.target_seq, args.indel_mode)

                # Process batches
                embeddings_list = []
                pseudo_likelihood_list = []
                sequences_list = []
                mutants_list = []
                
                mutant_index = 0
                with torch.no_grad():
                    for batch in tqdm.tqdm(dataloader, desc="Embedding sequences", total=len(dataloader)):
                        if args.model_type=="MSA_Transformer":
                            processed_batch = process_MSA_Transformer_batch(
                                batch=batch,
                                model=model,
                                alphabet=alphabet,
                                max_positions=args.max_positions,
                                MSA_start_position=args.MSA_start,
                                MSA_end_position=args.MSA_end,
                                MSA_weights=MSA_weights,
                                MSA_sequences=MSA_sequences,
                                num_MSA_sequences=args.num_MSA_sequences,
                                eval_mode=True,
                                start_idx=1,
                                long_sequences_slicing_method=args.long_sequences_slicing_method,
                                indel_mode=args.indel_mode,
                                fast_MSA_mode=args.fast_MSA_mode,
                                clustalomega_path=args.path_to_clustalomega,num_extra_tokens=1
                            )
                            embeddings, pseudo_ll, processed_batch = get_embeddings_MSA_Transformer(
                                model=model,
                                processed_batch=processed_batch,
                                alphabet_size=alphabet_size,
                                fast_MSA_mode=False
                            )
                            mutant, sequence = zip(*processed_batch['mutant_mutated_seq_pairs'])
                        elif args.model_type.startswith("ESM"):
                            processed_batch = process_ESM_batch(
                                batch=batch,
                                model=model,
                                alphabet=alphabet,
                                max_positions=args.max_positions,
                                long_sequences_slicing_method=args.long_sequences_slicing_method,
                                eval_mode=True,
                                start_idx=1,
                                num_extra_tokens=2
                            )
                            embeddings, pseudo_ll = get_embeddings_ESM(
                                model=model,
                                model_type=args.model_type,
                                processed_batch=processed_batch,
                                alphabet_size=alphabet_size
                            )
                            mutant, sequence = zip(*processed_batch['mutant_mutated_seq_pairs'])
                        elif args.model_type=="Tranception":
                            processed_batch = process_Tranception_batch(
                                batch=batch,
                                model=model
                            )
                            embeddings, pseudo_ll = get_embeddings_Tranception(
                                model=model,
                                processed_batch=processed_batch
                            )
                            full_batch_length = len(processed_batch['input_ids'])
                            sequence = np.array(df_chunk['mutated_sequence'][mutant_index:mutant_index+full_batch_length])
                            mutant = np.array(df_chunk['mutant'][mutant_index:mutant_index+full_batch_length])
                            mutant_index += full_batch_length

                        # Process embeddings
                        assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (batch_size,seq_len,embedding_dim)"
                        B,L,D = embeddings.shape
                        embeddings = [embedding.view(1,L,D).cpu() for embedding in embeddings]
                        if args.half_precision:
                            embeddings = [embedding.half() for embedding in embeddings]
                        
                        embeddings_list += embeddings
                        pseudo_likelihood_list.append(list(pseudo_ll))
                        sequences_list.append(list(sequence))
                        mutants_list.append(list(mutant))

                # Then in the embedding processing section, replace:
                if args.indel_mode or len(set(len(seq) for seq in df_chunk['mutated_sequence'])) > 1:
                    # Use global max length instead of chunk max length
                    embedding_dim = embeddings_list[0].shape[-1]
                    num_embeddings = len(embeddings_list)
                    
                    # Create padded tensor using global max length
                    embeddings = torch.zeros(num_embeddings, global_max_length, embedding_dim)
                    if args.half_precision:
                        embeddings = embeddings.half()
                        
                    # Copy each embedding into padded tensor
                    for emb_index, embedding in enumerate(embeddings_list):
                        embedding_len = embedding.shape[1]
                        embeddings[emb_index, :embedding_len] = embedding.squeeze(0)
                else:
                    embeddings = torch.cat(embeddings_list, dim=0)

                assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (num_mutated_seqs,seq_len,embedding_dim)"

                assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (num_mutated_seqs,seq_len,embedding_dim)"
                pseudo_likelihoods = torch.tensor(pseudo_likelihood_list)
                sequences = sum(sequences_list, [])
                mutants = sum(mutants_list, [])

                # Save chunk results
                chunk_embeddings_dict = {
                    'embeddings': embeddings,
                    'pseudo_likelihoods': pseudo_likelihoods,
                    'sequences': sequences,
                    'mutants': mutants
                }
                save_chunk_results(output_embeddings_path, chunk_idx, chunk_embeddings_dict)

                # Clear memory
                del df_chunk, embeddings_list, pseudo_likelihood_list, sequences_list, mutants_list
                del embeddings, pseudo_likelihoods, sequences, mutants, chunk_embeddings_dict
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                # Save progress point
                with open('embedding_progress.txt', 'w') as f:
                    f.write(str(chunk_start))
                raise e

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

    finally:
        # Clear any remaining memory
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)