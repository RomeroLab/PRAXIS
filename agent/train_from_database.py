import os,gc,sys
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import tqdm
import json
import argparse
import itertools
from datasets import Dataset
from datetime import datetime
from proteinnpt.utils.model_utils import Trainer
from proteinnpt.utils.data_utils import standardize
from proteinnpt.utils.esm.pretrained import load_model_and_alphabet

from utils import setup_config_and_paths, connect_db, get_train_database, score_mutated_sequences, data_cleanup_for_model_input, add_back_embedding, generate_mutation_string, get_segmented_sequence, sample_chunk, add_end_token_to_generated_sequence, fetch_previously_acquired_sequences, get_embeddings_ESM, process_ESM_batch, get_ESM_dataloader, convert_fragment_df_to_dict
from proteinnpt.proteinnpt.model import ProteinNPTModel
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def calculate_enzyme_specificity(target, off1, off2, sigma_target, sigma_off1, sigma_off2):
    """
    L1-based specificity: S = T^2 / (T + O1 + O2)
    This is target activity weighted by the fraction of total activity on the target.
    Propagates uncertainties via partial derivatives.
    Assumes inputs are pandas Series or numpy arrays.
    """
    target_val = np.array(target, dtype=float)
    off1_val = np.array(off1, dtype=float)
    off2_val = np.array(off2, dtype=float)
    sigma_target_val = np.array(sigma_target, dtype=float)
    sigma_off1_val = np.array(sigma_off1, dtype=float)
    sigma_off2_val = np.array(sigma_off2, dtype=float)

    epsilon = 1e-12
    denom = target_val + off1_val + off2_val + epsilon

    specificity = (target_val**2) / denom

    sigma_specificity = np.zeros_like(specificity)

    valid = denom > epsilon
    if np.any(valid):
        T_v = target_val[valid]
        O1_v = off1_val[valid]
        O2_v = off2_val[valid]
        sT_v = sigma_target_val[valid]
        sO1_v = sigma_off1_val[valid]
        sO2_v = sigma_off2_val[valid]
        D_v = denom[valid]
        D_sq = D_v**2

        # Partial derivatives of S = T^2 / (T + O1 + O2):
        # dS/dT  = T(T + 2*O1 + 2*O2) / (T + O1 + O2)^2
        # dS/dO1 = -T^2 / (T + O1 + O2)^2
        # dS/dO2 = -T^2 / (T + O1 + O2)^2
        dsdt_v = (T_v * (T_v + 2 * O1_v + 2 * O2_v)) / D_sq
        dsdo1_v = -(T_v**2) / D_sq
        dsdo2_v = -(T_v**2) / D_sq

        sigma_specificity[valid] = np.sqrt(
            (dsdt_v * sT_v)**2 +
            (dsdo1_v * sO1_v)**2 +
            (dsdo2_v * sO2_v)**2
        )

    return specificity, sigma_specificity


def sample_mutants(sequence_segments, acquired_sequences_database_path, acquired_sequences_table_name, model=None, num_samples=1000, mode="all_combinations", training_data_df=None, target_processing=None, phenotype_name="kcat", sampling_temperature=1.0, conditional_sampling_batch_size=30, conditional_sampling_top_quantile_selection=0.75, starting_sequence=None, verbose=False):
    """
    Returns a fixed number of mutated_sequences, conditionally sampled from the model.
    Assumes sequence_segments is a dict where keys are segment indices (0-indexed) and values are eligible segments at this index position.
    conditional_sampling_batch_size: 
        - in conditional sampling, this is the number of samples that we obtain at each iteration
        - the idea is to randomly sample conditional_sampling_batch_size points from the training data, mask 1 chunk at a time, and sample a replacement chunk.
        - once we have done num_samples // conditional_sampling_batch_size of these cycles, we would have generated num_samples new samples
    """
    if mode=="all_combinations":
        segment_keys = sorted(sequence_segments)
        segments_in_order = [sequence_segments[key] for key in segment_keys]
        all_combinations = itertools.product(*segments_in_order)
        segmented_sequences = [list(combination) for combination in all_combinations]
        generated_sequences = [''.join(segmented_sequence) for segmented_sequence in segmented_sequences]
    
    elif mode=="sample_at_random":
        segmented_sequences = []
        generated_sequences = []
        for _ in range(num_samples):
            sampled_combination = [random.choice(sequence_segments[key]) for key in sorted(sequence_segments)]
            segmented_sequences.append(sampled_combination)
            generated_sequences.append(''.join(sampled_combination))
    
    elif mode=="conditional_sampling":
        generated_sequences = []
        segmented_sequences = []
        num_chunks = len(list(sequence_segments.keys()))
            
        # Select high-performing sequences to conditionnally sample from
        phenotype_max_value = training_data_df[phenotype_name].max()
        top_quantile_threshold = training_data_df[phenotype_name].quantile(conditional_sampling_top_quantile_selection) #By default the threshold is set to top quartile
        filtered_data = training_data_df[training_data_df[phenotype_name] >= top_quantile_threshold]
        
        if len(filtered_data) > conditional_sampling_batch_size:
            conditional_sampling_set = filtered_data.sample(conditional_sampling_batch_size).copy().reset_index(drop=True)
        elif conditional_sampling_batch_size >= len(filtered_data) and len(training_data_df) > conditional_sampling_batch_size:
            conditional_sampling_batch_size = len(filtered_data)
            conditional_sampling_set = filtered_data.copy().reset_index(drop=True)
        else:
            conditional_sampling_batch_size = len(training_data_df)
            conditional_sampling_set = training_data_df.copy().reset_index(drop=True)
        
        conditional_sampling_set[phenotype_name] = phenotype_max_value

        training_data_df = data_cleanup_for_model_input(training_data_df, del_vars=False)
        # Sample and Mask
        for _ in tqdm.tqdm(range(num_samples // conditional_sampling_batch_size), "Conditional sampling of new sequences to score"):
            random_sequence_masking_order = [random.sample(range(num_chunks), num_chunks) for _ in range(conditional_sampling_batch_size)] #sampling w/o replacement a chunk order to mask for each sequence in the conditional_sampling_set
            for sampling_iteration_index in range(num_chunks):
                masked_sequences = []
                masked_segmented_sequences = []
                for row_index, row in conditional_sampling_set.iterrows():
                    # Mask the sequence at the chunk indexed by the random_sequence_masking_order
                    masked_segment_index = random_sequence_masking_order[row_index][sampling_iteration_index]
                    original_segmented_sequence = get_segmented_sequence(row) #['segmented_sequence']
                    # Mask the relevant sequence segment indexed by random_decoding_order
                    len_masked_segment = len(original_segmented_sequence[masked_segment_index])
                    masked_segmented_sequence = original_segmented_sequence.copy()
                    masked_segmented_sequence[masked_segment_index] = '<mask>' * len_masked_segment
                    masked_sequence = ''.join(masked_segmented_sequence)
                    masked_sequences.append(masked_sequence)
                    masked_segmented_sequences.append(masked_segmented_sequence)

                # Get predictions and select segment
                test_data_df = pd.DataFrame({
                    'sequence': masked_sequences,
                    'segmented_sequence': masked_segmented_sequences
                })
                test_data_df['mutant'] = test_data_df["sequence"].apply(lambda x: generate_mutation_string(starting_sequence, x))
                test_data_df = data_cleanup_for_model_input(test_data_df)

                add_back_embedding(model)
                logits = score_mutated_sequences(
                    model=model, 
                    test_data_df=test_data_df, 
                    training_data_df=training_data_df, 
                    target_processing=target_processing,
                    return_uncertainty=False,
                    verbose=False
                )["logits_protein_sequence"]
                logits = [torch.tensor(logit) for logit in logits]
                logits = pad_sequence(logits, batch_first=True, padding_value=0)
                log_probas = torch.log_softmax(logits / sampling_temperature, dim=-1)  # Apply temperature
                
                for row_index, row in conditional_sampling_set.iterrows():
                    masked_segment_index = random_sequence_masking_order[row_index][sampling_iteration_index]
                    
                    start_index = len(''.join(get_segmented_sequence(row)[:masked_segment_index]))
                    end_index = start_index + len(get_segmented_sequence(row)[masked_segment_index])
                    masked_segment_log_probas = log_probas[row_index, start_index:end_index, :]
                    best_chunk = sample_chunk(model, sequence_segments, masked_segment_index, masked_segment_log_probas)
                    
                    # Insert the best chunk considering its length
                    new_sampled_segmented_sequence = get_segmented_sequence(row).copy()
                    new_sampled_segmented_sequence[masked_segment_index] = best_chunk
                    new_sampled_sequence = ''.join(new_sampled_segmented_sequence)
                    new_sampled_sequence, new_sampled_segmented_sequence = add_end_token_to_generated_sequence(new_sampled_sequence, new_sampled_segmented_sequence)

                    generated_sequences.append(new_sampled_sequence)
                    segmented_sequences.append(new_sampled_segmented_sequence)
            
                    # Check if we have enough samples
                    if len(generated_sequences) >= num_samples:
                        break

    # Filter out sequences that have been previously acquired
    acquired_sequences = fetch_previously_acquired_sequences(acquired_sequences_database_path, acquired_sequences_table_name)
    acquired_sequences = [seq.replace('*', '') for seq in acquired_sequences] #remove the end '*' token
    if verbose: print("previously acquired sequences (count = {})".format(len(acquired_sequences)))
    unique_generated_sequences = set()
    filtered_generated_sequences = []
    filtered_segmented_sequences = []
    for sequence_index, sequence in enumerate(generated_sequences):
        if sequence.replace('*', '') not in acquired_sequences and sequence not in unique_generated_sequences:
            filtered_generated_sequences.append(sequence)
            filtered_segmented_sequences.append(segmented_sequences[sequence_index])
            unique_generated_sequences.add(sequence)
    if verbose: print("filtered_generated_sequences (count = {})".format(len(filtered_generated_sequences)))
    return filtered_generated_sequences, filtered_segmented_sequences


def get_main_target_from_config(target_config):
    for target in target_config:
        if target_config[target]["main_target"]:
            return target
    return None

def train_model(args, tablespace_location, database_name, table_name, train_data_df, target_processing, model_checkpoint_folder=None, model_name="PNPT_model", embeddings_location=None, verbose=False):
    """
    Train and return a PNPT model on an input training dataframe.
    """
    args.sequence_embeddings_location = embeddings_location
    
    for column in train_data_df.columns:
        if column in train_data_df.columns and train_data_df[column].isnull().any():
            print(f"Fixing null values in {column}")
            train_data_df[column] = train_data_df[column].fillna(0.0)

    for column in train_data_df.columns:
        null_count = train_data_df[column].isnull().sum()
        if null_count > 0:
            print(f"Warning: Found {null_count} null values in column {column}")
            print("Sample of rows with null values:")
            print(train_data_df[train_data_df[column].isnull()])
            raise ValueError(f"Null values found in column {column}")
    
    if verbose: 
        print("Size of training data:", len(train_data_df))
        print("Training data columns:", train_data_df.columns.tolist())
        print("Sample of training data:")
        print(train_data_df.head())
    
    if model_checkpoint_folder is not None:
        train_data_df.to_csv(model_checkpoint_folder + os.sep + model_name + '_training_data.csv', index=False)
    
    train_data_df_cleaned = data_cleanup_for_model_input(train_data_df.copy())
    
    if args.aa_embeddings=="MSA_Transformer":
        alphabet = Alphabet.from_architecture("msa_transformer")
    elif args.aa_embeddings=="Tranception":
        alphabet = get_tranception_tokenizer()
    else:
        alphabet = Alphabet.from_architecture("ESM-1b")

    if args.model_type=="ProteinNPT":
        model = ProteinNPTModel(args, alphabet)
    else:
        print("Model architecture requested is not currently available.")
        sys.exit(0)
    
    if args.frozen_embedding_parameters and (args.aa_embeddings in ["MSA_Transformer", "Tranception"] or args.aa_embeddings.startswith("ESM")):
        for para in model.aa_embedding.parameters():
            para.requires_grad = False
            
    trainer = Trainer(
        model=model,
        args=args,
        train_data=Dataset.from_pandas(train_data_df_cleaned), 
        val_data=None,
        target_processing=target_processing
    )
    
    trainer_final_status = trainer.train()
    
    if model_checkpoint_folder is not None:
        save_path = model_checkpoint_folder + os.sep + model_name + os.sep + 'final'
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        if hasattr(model, 'aa_embedding') and args.frozen_embedding_parameters: 
            del model.aa_embedding 
        torch.save({
            'args': args,
            'final_training_step': trainer_final_status['total_training_steps'],
            'state_dict': model.state_dict(),
            }, 
            save_path + os.sep + 'checkpoint.t7'
        )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, target_processing

def parse_arguments():
    parser = argparse.ArgumentParser(description='Self-driving')
    #Common to all agents - General params
    parser.add_argument('--data_location', type=str, help='Path to main PNPT data files')
    parser.add_argument('--sequence_segments_location', type=str, help='Path to csv file that contains sequence segments')
    parser.add_argument('--starting_sequence', type=str, help='Path to starting sequence (eg., wild type)')
    parser.add_argument('--model_config_location', type=str, help='Path to config of main model')
    parser.add_argument('--uncertainty_method', default="MC_dropout", type=str, help='Approach to compute model uncertainty')
    parser.add_argument('--num_MC_dropout_samples', default="MC_dropout", type=int, help='Number of MC dropout samples for uncertainty estimate')
    parser.add_argument('--embeddings_location', default=None, type=str, help='Path to location where we store all precomputed sequence embeddings')
    parser.add_argument('--zero_shot_fitness_predictions_location', default=None, type=str, help='Path to location where we store all zero-shot fitness predictions')
    parser.add_argument('--model_checkpoint_folder', type=str, help='Path to folder in which to score the embeddings for trained models')
    parser.add_argument('--tablespace_location', type=str, help='Path to location where underlying database should be stored')
    parser.add_argument('--database_name', type=str, help='Name of database to be shared across all agents')
    parser.add_argument('--table_name', type=str, help='Name of table to be shared across all agents')
    parser.add_argument('--selection_method', default="top_clusters", type=str, help='Method to select top sequences to aquire (used with conditional sampling only)')
    parser.add_argument('--ESM_location', type=str, help='Path to ESM2 (650M) location -- used to get sequence embeddings used in clustering of acquisition function')
    parser.add_argument('--target_to_index_mapping', type=str, help='Path to config file mapping target name to ECL target index')
    #Common to all agents - Bayesian optimization params
    parser.add_argument('--sampling_mode', default="conditional_sampling", type=str, help='Approach to select sequences to score at each round [conditional_sampling|all_combinations|sample_at_random]')
    parser.add_argument('--num_acquisitions', type=int, help='Total number of data acquisitions (ie., number of Bayesian Optimization rounds)')
    parser.add_argument('--acquisition_batch', type=int, default=3, help='Number of datapoints to acquire at each round (per agent)')
    parser.add_argument('--num_samples_per_iteration', type=int, help='If not exhaustively scoring all remaining sequences at each round, this is the number of sequences to sample and score')
    parser.add_argument('--acquisition_method', default="ucb", type=str, help='Acquisition function to use during Bayesian Optimization')
    parser.add_argument('--activity_threshold', default=0.001, type=float, help='Activity threshold value beyond which sequence is deemed active')
    #Agent-specific
    parser.add_argument('--target_config_location', type=str, help='Path to target config file')
    parser.add_argument('--model_name_suffix', type=str, help='Model name suffix')
    
    #Legacy params
    parser.add_argument('--MSA_data_folder', default=None, type=str, help='Path to main PNPT data files')
    parser.add_argument('--MSA_weight_data_folder', default=None, type=str, help='Path to main PNPT data files')
    parser.add_argument('--path_to_hhfilter', default=None, type=str, help='Path to main PNPT data files')
    parser.add_argument('--num_data_loaders_workers', default=0, type=int, help='Number of workers to use to fetch and load data in memory')
    #Training & Eval parameters
    parser.add_argument('--num_total_training_steps', default=None, type=int, help='Number of total training steps')
    parser.add_argument('--do_not_save_model_checkpoint', action='store_true', help='Whether to save model checkpoint')
    parser.add_argument('--max_tokens_per_msa', default=2**14, type=int, help='Used during inference to batch attention computations in a single forward pass. This allows increased input sizes with less memory.')
    parser.add_argument('--early_stopping_patience', default=5, type=int, help='Number of consecutive evals for which the loss has to not go below the min value to call early stopping (if None, no early stopping)')
    parser.add_argument('--max_learning_rate', default=3e-4, type=float, help='Max learning rate after warmup')
    parser.add_argument('--min_learning_rate', default=1e-5, type=float, help='Min learning rate post warmup and cosine decline')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Term added to the denominator to improve numerical stability in AdamW')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing parameter in the MLM loss')
    parser.add_argument('--grad_norm_clip', default=1.0, type=float, help='Maximum gradient value above which we do gradient clipping')
    parser.add_argument('--fine_tune_model_embedding_parameters', action='store_true', help='Whether to fine tune the model providing protein sequence embeddings')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to log runs in wandb')
    parser.add_argument('--use_validation_set', action='store_true', help='Whether to use a validation set during training [If yes, we will stop training based on CV loss and patience param. Train until the end otherwise]')
    parser.add_argument('--training_fp16', action='store_true', help='Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.')
    parser.add_argument('--indel_mode', action='store_true', help='indel mode')
    
    return parser.parse_args()

if __name__ == "__main__":
    print(f"Running on GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    args = parse_arguments()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args = setup_config_and_paths(args)
    args.save_model_checkpoint = not args.do_not_save_model_checkpoint
    args.frozen_embedding_parameters = not args.fine_tune_model_embedding_parameters
    
    # Override checkpoint folder to save to a 'redo' directory
    args.model_checkpoint_folder = os.path.join(args.data_location, 'model_checkpoints_redo')

    # Load sequence segments
    sequence_segments = convert_fragment_df_to_dict(pd.read_csv(args.sequence_segments_location, index_col=0, low_memory=False))

    # Determine main phenotype
    target_config = json.load(open(args.target_config_location))
    phenotype_name = get_main_target_from_config(target_config)
    assert phenotype_name is not None, "Error identifying main target to optimize from config file"
    
    # Load the entire dataset from the database
    print("Loading data from database...")
    full_dataset_df, initial_target_processing = get_train_database(args, args.tablespace_location, args.database_name, args.table_name)
    print(f"Loaded {len(full_dataset_df)} sequences from the database.")
    
    if args.starting_sequence and os.path.exists(args.starting_sequence):
        with open(args.starting_sequence, 'r') as f:
            starting_sequence_str = f.read().strip()
    else:
        starting_sequence_str = ""
        print("Warning: Starting sequence file not found or not provided. Mutant strings may not be generated correctly.")

    full_dataset_df['mutant'] = full_dataset_df["sequence"].apply(lambda x: generate_mutation_string(starting_sequence_str, x))
    
    # Group by 'created_at' timestamp
    full_dataset_df['created_at'] = pd.to_datetime(full_dataset_df['created_at'])
    timestamp_groups = sorted(full_dataset_df['created_at'].unique())
    
    print(f"Found {len(timestamp_groups)} unique timestamps to iterate through for training.")

    cumulative_train_df = pd.DataFrame()
    
    for iteration_idx, timestamp in enumerate(timestamp_groups):
        print(f"\n--- Iteration {iteration_idx}: Processing timestamp {timestamp} ---")

        # Get the data for the current timestamp
        current_timestamp_df = full_dataset_df[full_dataset_df['created_at'] == timestamp]
        
        # Add the current group to the cumulative training data
        cumulative_train_df = pd.concat([cumulative_train_df, current_timestamp_df], ignore_index=True)
        num_train_samples = len(cumulative_train_df)

        print(f"Training on {num_train_samples} sequences (added {len(current_timestamp_df)} from this timestamp).")

        # Set the training dataframe for this round
        train_df = cumulative_train_df.copy()

        # Define model name for this round, mirroring self_driving.py
        model_name_for_round = '_'.join([args.model_name_suffix, 'round', str(iteration_idx)])

        # Train the model
        model, target_processing = train_model(
            args=args,
            tablespace_location=args.tablespace_location,
            database_name=args.database_name,
            table_name=args.table_name,
            train_data_df=train_df,
            target_processing=initial_target_processing,
            model_checkpoint_folder=args.model_checkpoint_folder,
            model_name=model_name_for_round,
            embeddings_location=args.embeddings_location,
            verbose=True
        )

        # ---- REPLICATION OF self_driving.py SEQUENCE GENERATION AND SCORING ----
        print(f"Generating {args.num_samples_per_iteration} new sequences to score...")

        # Determine conditioning phenotype for sampling
        conditioning_phenotype = phenotype_name
        if args.acquisition_method.startswith("spec"):
            if args.acquisition_method == "spec1":
                spec, _ = calculate_enzyme_specificity(
                    train_df["a1"], train_df["a2"], train_df["a3"],
                    np.zeros(len(train_df)), np.zeros(len(train_df)), np.zeros(len(train_df))
                )
                train_df["exp_spec_s1"] = spec
                conditioning_phenotype = "exp_spec_s1"
            elif args.acquisition_method == "spec2":
                spec, _ = calculate_enzyme_specificity(
                    train_df["a2"], train_df["a1"], train_df["a3"],
                    np.zeros(len(train_df)), np.zeros(len(train_df)), np.zeros(len(train_df))
                )
                train_df["exp_spec_s2"] = spec
                conditioning_phenotype = "exp_spec_s2"
            elif args.acquisition_method == "spec3":
                spec, _ = calculate_enzyme_specificity(
                    train_df["a3"], train_df["a1"], train_df["a2"],
                    np.zeros(len(train_df)), np.zeros(len(train_df)), np.zeros(len(train_df))
                )
                train_df["exp_spec_s3"] = spec
                conditioning_phenotype = "exp_spec_s3"

        # Generate new sequences using the trained model
        sampled_sequences, sampled_sequences_segmented = sample_mutants(
            sequence_segments=sequence_segments, 
            acquired_sequences_database_path=os.path.join(args.tablespace_location, args.database_name), 
            acquired_sequences_table_name=args.table_name,
            model=model, 
            num_samples=args.num_samples_per_iteration, 
            mode=args.sampling_mode, 
            training_data_df=train_df, 
            target_processing=target_processing, 
            phenotype_name=conditioning_phenotype,
            starting_sequence=starting_sequence_str,
            verbose=True
        )

        if sampled_sequences:
            # Create test dataframe from newly generated sequences
            test_data_df = pd.DataFrame({
                'sequence': sampled_sequences,
                'segmented_sequence': sampled_sequences_segmented
            })
            test_data_df['mutant'] = test_data_df["sequence"].apply(lambda x: generate_mutation_string(starting_sequence_str, x))
            
            print(f"Scoring {len(test_data_df)} generated sequences...")
            
            # Prepare data for scoring
            train_df_cleaned = data_cleanup_for_model_input(train_df.copy())
            test_df_cleaned = data_cleanup_for_model_input(test_data_df.copy())
            
            add_back_embedding(model)
            
            # Score the generated sequences
            predictions_df = score_mutated_sequences(
                model=model,
                test_data_df=test_df_cleaned,
                training_data_df=train_df_cleaned,
                target_processing=target_processing,
                uncertainty_method=args.uncertainty_method,
                num_MC_dropout_samples=args.num_MC_dropout_samples,
                return_uncertainty=True,
                verbose=True
            )

            # Save predictions, mirroring the structure of self_driving.py
            predictions_folder = os.path.join(args.data_location, 'model_predictions_redo', args.model_name_suffix)
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)
            
            prediction_filename = '_'.join([args.model_name_suffix, 'round', str(iteration_idx)]) + '.csv'
            prediction_filepath = os.path.join(predictions_folder, prediction_filename)
            predictions_df.to_csv(prediction_filepath, index=False)
            print(f"Saved predictions to {prediction_filepath}")
        
        else:
            print("No new sequences were generated. Ending training cycle for this timestamp.")
        
        # Check if this was the last timestamp group
        if iteration_idx == len(timestamp_groups) - 1:
            print("All timestamps have been processed. Final iteration complete.")
            break

    print("\nIncremental training and prediction process finished.")
