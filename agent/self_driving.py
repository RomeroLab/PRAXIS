import os,gc,sys
import pandas as pd
import numpy as np
import paramiko
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import tqdm
import json
import argparse
import itertools
from sklearn.cluster import KMeans
from datasets import Dataset
from git import Repo
from datetime import datetime,time
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from phenotype_monitor import monitor_phenotype_data
import platform
import shutil
import requests

from utils import setup_config_and_paths, connect_db, create_table, add_sequence_data, get_train_database, score_mutated_sequences, convert_fragment_df_to_dict, data_cleanup_for_model_input, add_back_embedding, get_segmented_sequence, generate_mutation_string, sample_chunk, add_end_token_to_generated_sequence, fetch_previously_acquired_sequences, get_embeddings_ESM, process_ESM_batch, get_ESM_dataloader

# Re-export ConvBertLayer onto transformers so proteinnpt imports (see _compat_proteinnpt).
import _compat_proteinnpt  # noqa: F401
from proteinnpt.proteinnpt.model import ProteinNPTModel
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.esm.pretrained import load_model_and_alphabet
from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.model_utils import Trainer
from proteinnpt.utils.data_utils import standardize

import utils

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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

        print("\nDEBUG training_data_df:")
        print("Columns:", training_data_df.columns)
        print("Sample segmented_sequence:", training_data_df['segmented_sequence'].iloc[0])
        print("Type of segmented_sequence:", type(training_data_df['segmented_sequence'].iloc[0]))
        print("Length of segmented_sequence:", len(training_data_df['segmented_sequence'].iloc[0]))
            
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

        # Validate and print segment information
        print(f"Number of chunks: {num_chunks}")
        print(f"Available segment indices: {list(sequence_segments.keys())}")

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
                #test_data_df['mutant'] = test_data_df["sequence"].apply(lambda x: ':'.join([wt + str(i+1) + mut for i, (wt, mut) in enumerate(zip(starting_sequence, x)) if wt != mut]))
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
    #print("previously acquired sequences (count = {}): {}".format(len(acquired_sequences), acquired_sequences))
    if verbose: print("previously acquired sequences (count = {})".format(len(acquired_sequences)))
    unique_generated_sequences = set()
    filtered_generated_sequences = []
    filtered_segmented_sequences = []
    for sequence_index, sequence in enumerate(generated_sequences):
        if sequence.replace('*', '') not in acquired_sequences and sequence not in unique_generated_sequences:
            filtered_generated_sequences.append(sequence)
            filtered_segmented_sequences.append(segmented_sequences[sequence_index])
            unique_generated_sequences.add(sequence)
    #print("filtered_generated_sequences (count = {}): {}".format(len(filtered_generated_sequences), filtered_generated_sequences))
    if verbose: print("filtered_generated_sequences (count = {})".format(len(filtered_generated_sequences)))
    return filtered_generated_sequences, filtered_segmented_sequences


def train_model(args, tablespace_location, database_name, table_name, model_checkpoint_folder=None, model_name="PNPT_May4_thread0_step0", embeddings_location=None, verbose=False):
    """
    Train and return a PNPT model on an input training dataframe
    """
    args.sequence_embeddings_location = embeddings_location
    setup_config_and_paths(args)
    
    # Get training data
    train_data_df, target_processing = get_train_database(args, tablespace_location, database_name, table_name)


    #replace null values with 0.0
    for column in train_data_df.columns:
        if column in train_data_df.columns and train_data_df[column].isnull().any():
            print(f"Fixing null values in {column}")
            train_data_df[column] = train_data_df[column].fillna(0.0)

    # Validate data before training
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
    
    train_data_df = data_cleanup_for_model_input(train_data_df)
    
    # Initialize model and trainer
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
        train_data=Dataset.from_pandas(train_data_df), 
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
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, target_processing

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

    print(f"Calculated specificity: {specificity}")
    print(f"Calculated sigma_specificity: {sigma_specificity}")
    return specificity, sigma_specificity

def select_mutants_to_acquire(mutated_sequences_pred, phenotype_name="kcat", num_sequences_to_acquire=30, acquisition_method="ucb", ucb_alpha=1.0, selection_method="top_clusters", ESM_location=None, exclude_sequences=None):
    """
    Function to select mutants to acquire based on standard Bayesian Optimization acquisition functions.
    Only UCB is implemented for now.

    Parameters:
    - mutated_sequences_pred (DataFrame): DataFrame containing the columns 'sequence', 'prediction', 'uncertainty'.
    - num_sequences_to_acquire (int): Number of sequences to select for acquisition.
    - acquisition_method (str): Method to use for selecting sequences. Default is 'ucb' (Upper Confidence Bound).
    - selection_method (str): Method to select the sequences to acquire. Default is 'top_clusters' (ie. select top 30 clusters based on score_column, cluster these sequences, takes the best sequence by cluster)
    Returns:
    - List of sequences to acquire.
    """
    if acquisition_method.lower() == "ucb":
        score_column = 'ucb'
        # Calculate the Upper Confidence Bound for each sequence
        mutated_sequences_pred[score_column] = mutated_sequences_pred["predictions_avg_"+phenotype_name] + ucb_alpha * mutated_sequences_pred['uncertainty_'+phenotype_name]
        
    elif acquisition_method in ["spec1","spec2","spec3"]:
        a1 = mutated_sequences_pred["predictions_avg_a1"]
        a2 = mutated_sequences_pred["predictions_avg_a2"]
        a3 = mutated_sequences_pred["predictions_avg_a3"]
        sigma1 = mutated_sequences_pred['uncertainty_a1']
        sigma2 = mutated_sequences_pred['uncertainty_a2']
        sigma3 = mutated_sequences_pred['uncertainty_a3']

        if acquisition_method=="spec1":
            score_column = 'spec1_score'
            #calculate specificity WRT substrate 1
            spec1,sigma_spec1 = calculate_enzyme_specificity(a1, a2, a3, sigma1, sigma2, sigma3)
            mutated_sequences_pred[score_column] = spec1 + ucb_alpha * sigma_spec1

        elif acquisition_method == "spec2":
            score_column = 'spec2_score'
            #calculate specificity WRT substrate 2
            spec2,sigma_spec2 = calculate_enzyme_specificity(a2, a1, a3, sigma2, sigma1, sigma3)
            mutated_sequences_pred[score_column] = spec2 + ucb_alpha * sigma_spec2

        elif acquisition_method == "spec3":
            score_column = 'spec3_score'
            #calculate specificity WRT substrate 3
            spec3,sigma_spec3 = calculate_enzyme_specificity(a3, a1, a2, sigma3, sigma1, sigma2)
            mutated_sequences_pred[score_column] = spec3 + ucb_alpha * sigma_spec3

    else:
        raise ValueError("Unsupported acquisition method: {}".format(acquisition_method))

    if exclude_sequences:
        exclude_clean = {s.replace('*', '') for s in exclude_sequences}
        mutated_sequences_pred = mutated_sequences_pred[~mutated_sequences_pred['sequence'].isin(exclude_clean)].copy()

    if selection_method.lower()=="top_n":
        # Sort sequences based on the UCB scores in descending order
        mutated_sequences_pred.sort_values(by=score_column, ascending=False, inplace=True)
        
        # Select the top 'num_sequences_to_acquire' sequences
        mutated_sequences_to_acquire = mutated_sequences_pred.head(num_sequences_to_acquire)['sequence'].tolist()

    elif selection_method.lower()=="top_clusters":
        # Sort sequences based on the acquisition score in descending order, and select min(top 30, len(mutated_sequences_pred))
        num_sequences_to_cluster = min(30,len(mutated_sequences_pred))
        top_sequences = mutated_sequences_pred.nlargest(num_sequences_to_cluster, score_column)
        
        # Get mean ESM embeddings for these sequences
        model, alphabet = load_model_and_alphabet(ESM_location)
        top_seq_df = pd.DataFrame({'mutated_sequence': top_sequences['sequence'], 'mutant': top_sequences['mutant']})
        with torch.no_grad():
            dataloader = get_ESM_dataloader(top_seq_df, num_sequences_to_cluster) #We will just do 1 batch
            for batch in dataloader:
                processed_batch = process_ESM_batch(
                            batch = batch, 
                            model = model, 
                            alphabet = alphabet, 
                            max_positions = 1024, 
                            long_sequences_slicing_method = 'center', 
                            eval_mode = True, 
                            num_extra_tokens = 2
                            )
            embeddings = get_embeddings_ESM(model, model_type="ESM2_650M", processed_batch=processed_batch, alphabet_size=len(alphabet), return_pseudo_ll=False)[0]
        mean_embeddings = embeddings.mean(dim=1) # get mean-pooled embeddings across sequence length
        # Perform clustering using sequence embeddings
        num_clusters = min(num_sequences_to_acquire, len(top_sequences))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(np.stack(mean_embeddings.cpu().numpy()))

        # Select the best sequence from each cluster
        top_sequences['cluster'] = clusters
        best_sequences = top_sequences.groupby('cluster')[['sequence', score_column]].apply(lambda x: x.nlargest(1, score_column)).reset_index(drop=True)
        mutated_sequences_to_acquire=best_sequences['sequence'].tolist()
        
    return mutated_sequences_to_acquire

def transfer_file_via_http(local_path, server_url, target_dir):
    """
    Transfer a file to a Windows computer using HTTP POST
    
    Parameters:
    local_path (str): Path of file on local machine to send
    server_url (str): URL of the server (e.g., http://LAB_HOST:8000/upload)
    """
    try:
        # Prepare the file for upload
        filename = os.path.basename(local_path)
        with open(local_path, 'rb') as f:
            files = {'file': (filename, f)}
            
            # Send the file
            response = requests.post(server_url, files=files)
            
            # Check the response
            if response.status_code == 200:
                print(f'Successfully transferred {local_path} to server')
                return True
            else:
                print(f'Error: Server returned status code {response.status_code}')
                print(f'Response: {response.text}')
                return False
        
    except Exception as e:
        print(f'Error during file transfer: {str(e)}')
        return False

def combine_segments_and_save(segmented_sequences, output_file):
    """
    Combine segments and write all sequences directly to output_file.
    """
    with open(output_file, 'w') as f:
        if isinstance(segmented_sequences, pd.Series):
            for segments in segmented_sequences:
                f.write(f"{''.join(segments)}\n")
        else:
            for segments in segmented_sequences:
                f.write(f"{''.join(segments)}\n")
    
    print(f"Wrote {len(segmented_sequences)} sequences to {output_file}")

def lab_acquisition(segmented_sequences):
    """
    Write combined sequences to file, transfer to lab server, and monitor for results.
    """
    output_file = "data/sequence_query.txt"
    phenotype_file = "data/plate_data/phenotype.json"
    combine_segments_and_save(segmented_sequences, output_file)
    
    # Lab automation upload endpoint. Set LAB_UPLOAD_URL to your environment (`environment/`) file_reciever host.
    server_url = os.environ.get("LAB_UPLOAD_URL", "http://LAB_HOST:8000/upload")
    target_dir = "sequence_queries"
    transfer_file_via_http(output_file, server_url, target_dir)
    
    if os.path.exists(phenotype_file):
        os.remove(phenotype_file)
        print(f"Deleted existing {phenotype_file} before new query results")

    phenotype_data, invalid_sequences = monitor_phenotype_data(phenotype_file)
    
    if phenotype_data is None:
        raise Exception("Failed to get phenotype data")
        
    return phenotype_data, invalid_sequences

def get_dict_from_lab_acquisition(segmented_sequences, assayed_target_name, target_to_index_mapping):
    """
    Get measurements for valid sequences and track invalid ones
    """
    phenotype_data, invalid_sequences = lab_acquisition(segmented_sequences)
    
    lab_dict = {}
    for segmented_sequence in segmented_sequences:
        seq = ''.join(segmented_sequence)
        # Skip invalid sequences
        if seq in invalid_sequences:
            continue
            
        lab_dict[seq] = {}
        for target in assayed_target_name:
            lab_dict[seq][target] = phenotype_data[seq][target_to_index_mapping[target]]
    return lab_dict, invalid_sequences

def get_zero_shot_fitness_predictions(sequences, zero_shot_fitness_location, zero_shot_prediction_name="ESM2_650M"):
    """
    Retrieve zero-shot fitness predictions for a given list of sequences and return them as a list,
    maintaining the order of the provided sequences.

    Args:
    sequences (list): A list of DNA or protein sequences for which predictions are needed.
    zero_shot_fitness_location (str): File path to the CSV containing the fitness predictions.
    zero_shot_prediction_name (str): The column name in the CSV file that contains the fitness predictions.

    Returns:
    list: A list of fitness predictions corresponding to the input sequences, maintaining input order.
    """
    # Clean sequences by removing '*' characters
    cleaned_sequences = [seq.replace('*', '') for seq in sequences]

    # Load the dataset from the specified CSV file
    df = pd.read_csv(zero_shot_fitness_location)
    df[zero_shot_prediction_name] = standardize(df[zero_shot_prediction_name])

    # Ensure that the DataFrame contains the necessary columns
    if 'mutated_sequence' not in df.columns or zero_shot_prediction_name not in df.columns:
        raise ValueError("The CSV file must contain 'mutated_sequence' and '{}' columns".format(zero_shot_prediction_name))

    # Create a dictionary from the DataFrame for quick lookup
    predictions_dict = pd.Series(df[zero_shot_prediction_name].values, index=df['mutated_sequence']).to_dict()

    zero_shot_dict = {}
    for seq in cleaned_sequences:
        zero_shot_dict[seq+"*"] = predictions_dict.get(seq, None)

    return zero_shot_dict

def acquire_sequences(sequences_df, target_names, target_to_index_mapping, threshold=0.001, zero_shot_fitness_location=None, zero_shot_prediction_name="ESM2_650M"):
    """
    Send sequences to the lab for phenotype measurement. Handles all agents' sequences in one batch.
    """
    print(f"Target names: {target_names}")
    
    assayed_target_name = [target for target in target_names if target != "zero_shot_fitness_predictions"]
    acquired_phenotypes = {target: {} for target in assayed_target_name}
    if 'f' not in acquired_phenotypes and 'f' in target_names:
        acquired_phenotypes['f'] = {}
        
    acquired_target_name = [target for target in assayed_target_name if target != "f"]

    sequences_to_acquire_df = sequences_df.copy()
    
    lab_dict, invalid_sequences = get_dict_from_lab_acquisition(
        sequences_to_acquire_df['segmented_sequence'], 
        acquired_target_name, 
        target_to_index_mapping
    )

    for seq in sequences_to_acquire_df['sequence']:
        if seq in invalid_sequences:
            continue
            
        max_value = 0
        for target in acquired_target_name:
            if seq in lab_dict and target in lab_dict[seq]:
                target_value = lab_dict[seq][target]
                acquired_phenotypes[target][seq] = target_value
                if target_value > max_value:
                    max_value = target_value
        
        f_value = int(max_value > threshold)
        acquired_phenotypes['f'][seq] = f_value
    
    if "zero_shot_fitness_predictions" in target_names:
        valid_sequences = [seq for seq in sequences_df['sequence'] if seq not in invalid_sequences]
        if valid_sequences:
            zero_shot_dict = get_zero_shot_fitness_predictions(
                valid_sequences, 
                zero_shot_fitness_location, 
                zero_shot_prediction_name
            )
            acquired_phenotypes["zero_shot_fitness_predictions"] = {}
            for seq, pred in zero_shot_dict.items():
                acquired_phenotypes["zero_shot_fitness_predictions"][seq] = pred
        
    print("Sample of acquired_phenotypes:")
    for target, seq_dict in acquired_phenotypes.items():
        if seq_dict:
            sample_key = next(iter(seq_dict))
            print(f"Target: {target}, First value: {seq_dict[sample_key]}, Type: {type(seq_dict[sample_key])}")
    
    return acquired_phenotypes, invalid_sequences

ACQUISITION_METHOD_TO_AGENT = {
    "spec1": "agent_a1",
    "spec2": "agent_a2",
    "spec3": "agent_a3",
    "ucb": "agent_ucb",
}

def get_agent_name(acquisition_method):
    return ACQUISITION_METHOD_TO_AGENT.get(acquisition_method, f"agent_{acquisition_method}")

def get_main_target_from_config(target_config):
    for target in target_config:
        if target_config[target]["main_target"]:
            return target
    return None

def derive_segmented_sequence(sequence, sequence_segments):
    """Derive segmented_sequence by greedy-matching known fragments at each position."""
    num_segments = len(sequence_segments)
    pos = 0
    segments = []
    for i in range(num_segments):
        matched = False
        for frag in sequence_segments[f'f{i}']:
            if sequence[pos:pos+len(frag)] == frag:
                segments.append(frag)
                pos += len(frag)
                matched = True
                break
        if not matched:
            raise ValueError(f"No matching fragment at position f{i} (offset {pos}) for sequence {sequence[:50]}...")
    return segments

STARTING_SEQ_INDICES = {"ucb": [0], "spec1": [0, 1], "spec2": [2, 3], "spec3": [4, 5]}

def get_conditioning_phenotype(acquisition_method, train_data_df, default_phenotype):
    """
    Compute the conditioning phenotype for a given acquisition method.
    Adds a derived specificity column to train_data_df (in-place) and returns its name.
    For non-specificity methods, returns the default phenotype name.
    """
    zeros = np.zeros(len(train_data_df))
    if acquisition_method == "spec1":
        spec, _ = calculate_enzyme_specificity(
            train_data_df["a1"], train_data_df["a2"], train_data_df["a3"],
            zeros, zeros, zeros
        )
        train_data_df["exp_spec_s1"] = spec
        return "exp_spec_s1"
    elif acquisition_method == "spec2":
        spec, _ = calculate_enzyme_specificity(
            train_data_df["a2"], train_data_df["a1"], train_data_df["a3"],
            zeros, zeros, zeros
        )
        train_data_df["exp_spec_s2"] = spec
        return "exp_spec_s2"
    elif acquisition_method == "spec3":
        spec, _ = calculate_enzyme_specificity(
            train_data_df["a3"], train_data_df["a1"], train_data_df["a2"],
            zeros, zeros, zeros
        )
        train_data_df["exp_spec_s3"] = spec
        return "exp_spec_s3"
    else:
        return default_phenotype

def optimize_starting_sequence(starting_sequence, args, sequence_segments=None, num_acquisitions=20, acquisition_batch=30, num_samples_per_iteration=1000, acquisition_methods=None, sampling_mode="conditional_sampling",
                                uncertainty_method="MC_dropout", num_MC_dropout_samples=10, embeddings_location=None, target_config_location="./proteinnpt/utils/target_configs/fitness.json", target_to_index_mapping=None,
                                model_name="PNPT_May4", model_checkpoint_folder='./', table_name="training_data", database_name="training_database", tablespace_location='./', selection_method="top_clusters", ESM_location=None,
                                seed_sequences=None, starting_round=0, verbose=False):
    """
    Single-GPU multi-agent Bayesian Optimization.
    Trains ONE shared ProteinNPT model per iteration, then each agent independently
    samples candidates (conditioned on its own phenotype), scores them with the shared
    model, and selects sequences via its own acquisition function. All agents' selected
    sequences are sent to the lab in a single batch.
    """
    if acquisition_methods is None:
        acquisition_methods = ["ucb"]
    
    # Setup database and targets
    target_to_index_mapping = json.load(open(args.target_to_index_mapping))
    target_config = json.load(open(target_config_location))
    target_names = [target_name for target_name in target_config]
    if args.zero_shot_fitness_predictions_location is not None:
        target_names += ['zero_shot_fitness_predictions']
    phenotype_name = get_main_target_from_config(target_config)
    assert phenotype_name is not None, "Error identifying main target to optimize from config file"
    
    connection = connect_db(tablespace_location, database_name)
    create_table(connection, table_name, target_names)
    
    num_segments = len(list(sequence_segments.keys()))

    skip_init = (seed_sequences is not None) or (starting_round > 0)

    if skip_init:
        # =================================================================
        # Resume mode: seed DB from file and/or skip parent initialization
        # =================================================================
        if seed_sequences is not None:
            if seed_sequences.endswith('.json'):
                with open(seed_sequences) as f:
                    phenotype_data = json.load(f)
                target_mapping = json.load(open(args.target_to_index_mapping))
                rows = []
                agent_names = [get_agent_name(m) for m in acquisition_methods]
                for i, (seq, data) in enumerate(phenotype_data.items()):
                    if not data.get('valid', True):
                        continue
                    measurements = data.get('measurements', [])
                    if not measurements:
                        continue
                    row = {'sequence': seq}
                    for target, idx in target_mapping.items():
                        if idx < len(measurements):
                            row[target] = measurements[idx]
                    row['f'] = int(max(measurements) > args.activity_threshold)
                    row['mutant'] = generate_mutation_string(starting_sequence, seq)
                    row['segmented_sequence'] = derive_segmented_sequence(seq, sequence_segments)
                    if starting_round > 0:
                        row['created_by'] = agent_names[(i // acquisition_batch) % len(agent_names)]
                    rows.append(row)
                seed_df = pd.DataFrame(rows)
                if args.zero_shot_fitness_predictions_location:
                    zs_dict = get_zero_shot_fitness_predictions(
                        list(seed_df['sequence']),
                        args.zero_shot_fitness_predictions_location
                    )
                    seed_df['zero_shot_fitness_predictions'] = seed_df['sequence'].map(zs_dict)
            else:
                seed_df = pd.read_csv(seed_sequences)
            if 'created_by' not in seed_df.columns:
                agent_names = [get_agent_name(m) for m in acquisition_methods]
                created_by = []
                if starting_round > 0:
                    for i in range(len(seed_df)):
                        created_by.append(agent_names[(i // acquisition_batch) % len(agent_names)])
                else:
                    num_parents = len(acquisition_methods) * acquisition_batch
                    for i in range(len(seed_df)):
                        if i < num_parents:
                            created_by.append('seed')
                        else:
                            j = i - num_parents
                            created_by.append(agent_names[(j // acquisition_batch) % len(agent_names)])
                seed_df['created_by'] = created_by
            add_sequence_data(connection, table_name, seed_df)
            print(f"Seeded {len(seed_df)} sequences from {seed_sequences}")

        pre_round = max(0, starting_round - 1)
        print(f"Resuming: training model on existing DB data (round_{pre_round}), BO loop starts at round {max(1, starting_round)}")
        model, target_processing = train_model(
            args=args,
            tablespace_location=tablespace_location,
            database_name=database_name,
            table_name=table_name,
            model_checkpoint_folder=model_checkpoint_folder,
            model_name='_'.join([model_name, f'round_{pre_round}']),
            embeddings_location=embeddings_location,
            verbose=verbose
        )
    else:
        # =================================================================
        # Normal init: acquire starting sequences for ALL agents in one batch
        # Build parent sequences directly from sequence_segments fragments (p1-p6)
        # =================================================================
        init_segmented = []
        init_agent_names = []
        seen_seqs = set()
        for acq_method in acquisition_methods:
            indices = STARTING_SEQ_INDICES.get(acq_method, [0])
            if len(sequence_segments['f0']) < 4:
                indices = [0]  # test mode: single parent only
            for idx in indices:
                seg = [sequence_segments[f'f{i}'][idx] for i in range(num_segments)]
                seq_key = ''.join(seg)
                if seq_key not in seen_seqs:
                    init_segmented.append(seg)
                    init_agent_names.append(get_agent_name(acq_method))
                    seen_seqs.add(seq_key)

        print(f"Initializing with {len(init_segmented)} starting sequence(s) for agents: {init_agent_names}")

        init_sequences = [''.join(s) for s in init_segmented]
        training_df = pd.DataFrame({
            'sequence': init_sequences,
            'segmented_sequence': init_segmented
        })
        training_df['mutant'] = training_df["sequence"].apply(
            lambda x: ':'.join([wt + str(i+1) + mut for i, (wt, mut) in enumerate(zip(starting_sequence, x)) if wt != mut])
        )

        acquisition_dict, invalid_sequences = acquire_sequences(
            sequences_df=training_df,
            target_names=target_names,
            target_to_index_mapping=target_to_index_mapping,
            threshold=args.activity_threshold,
            zero_shot_fitness_location=args.zero_shot_fitness_predictions_location
        )

        if invalid_sequences:
            print(f"Removing {len(invalid_sequences)} invalid sequences from training data")
            training_df = training_df[~training_df['sequence'].isin(invalid_sequences)]

        for target, seq_values in acquisition_dict.items():
            print(f"Adding target {target} with values: {list(seq_values.values())[:3]}")
            training_df[target] = training_df['sequence'].map(seq_values)

        if not training_df.empty:
            training_df['created_by'] = init_agent_names[:len(training_df)]
            add_sequence_data(connection, table_name, training_df)

        # =====================================================================
        # Train initial shared model
        # =====================================================================
        model, target_processing = train_model(
            args=args,
            tablespace_location=tablespace_location,
            database_name=database_name,
            table_name=table_name,
            model_checkpoint_folder=model_checkpoint_folder,
            model_name='_'.join([model_name, 'round_0']),
            embeddings_location=embeddings_location,
            verbose=verbose
        )

    # =========================================================================
    # Main Bayesian Optimization loop
    # =========================================================================
    loop_start = max(1, starting_round) if skip_init else 1
    for iteration_idx in range(loop_start, num_acquisitions+1):
        print(f"========== STARTING ITER {iteration_idx}/{num_acquisitions} ==========")
        train_data_df, target_processing = get_train_database(args, tablespace_location, database_name, table_name)
        add_back_embedding(model)
        
        # --- Per-agent: sample, score, select ---
        all_agent_selections = []
        already_selected_sequences = set()

        for acq_method in acquisition_methods:
            agent_name = get_agent_name(acq_method)
            print(f"  [{agent_name}] ({acq_method}): sampling and scoring candidates...")
            
            agent_train_df = train_data_df.copy()
            conditioning_phenotype = get_conditioning_phenotype(acq_method, agent_train_df, phenotype_name)
            
            sampled_sequences, sampled_sequences_segmented = sample_mutants(
                sequence_segments=sequence_segments,
                acquired_sequences_database_path=tablespace_location + os.sep + database_name,
                acquired_sequences_table_name=table_name,
                model=model,
                num_samples=num_samples_per_iteration,
                mode=sampling_mode,
                training_data_df=agent_train_df,
                target_processing=target_processing,
                phenotype_name=conditioning_phenotype,
                starting_sequence=starting_sequence,
                verbose=verbose
            )
            
            test_data_df = pd.DataFrame({
                'sequence': sampled_sequences,
                'segmented_sequence': sampled_sequences_segmented
            })
            test_data_df['mutant'] = test_data_df["sequence"].apply(
                lambda x: ':'.join([wt + str(i+1) + mut for i, (wt, mut) in enumerate(zip(starting_sequence, x)) if wt != mut])
            )
            scoring_train_df = data_cleanup_for_model_input(agent_train_df)
            test_data_df = data_cleanup_for_model_input(test_data_df)
            
            add_back_embedding(model)
            sampled_sequences_pred = score_mutated_sequences(
                model=model,
                test_data_df=test_data_df,
                training_data_df=scoring_train_df,
                target_processing=target_processing,
                uncertainty_method=uncertainty_method,
                num_MC_dropout_samples=num_MC_dropout_samples,
                return_uncertainty=True
            )
            
            pred_dir = os.path.join(args.data_location, 'model_predictions', model_name)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            sampled_sequences_pred.to_csv(
                os.path.join(pred_dir, f'{model_name}_round_{iteration_idx}_{agent_name}.csv')
            )
            
            sequences_to_acquire = select_mutants_to_acquire(
                mutated_sequences_pred=sampled_sequences_pred,
                phenotype_name=conditioning_phenotype,
                num_sequences_to_acquire=acquisition_batch,
                acquisition_method=acq_method,
                selection_method=selection_method,
                ESM_location=ESM_location,
                exclude_sequences=already_selected_sequences
            )
            sequences_to_acquire = [seq + '*' for seq in sequences_to_acquire]
            already_selected_sequences.update(sequences_to_acquire)
            indices = [sampled_sequences.index(seq) for seq in sequences_to_acquire if seq in sampled_sequences]
            segs = [sampled_sequences_segmented[i] for i in indices]
            
            print(f"  [{agent_name}] selected {len(sequences_to_acquire)} sequence(s)")
            all_agent_selections.append((agent_name, sequences_to_acquire, segs))
        
        # --- Combine all agents' selections into one lab batch ---
        combined_sequences = []
        combined_segmented = []
        combined_agents = []
        for agent_name, seqs, segs in all_agent_selections:
            combined_sequences.extend(seqs)
            combined_segmented.extend(segs)
            combined_agents.extend([agent_name] * len(seqs))
        
        combined_df = pd.DataFrame({
            'sequence': combined_sequences,
            'segmented_sequence': combined_segmented,
            'created_by': combined_agents
        })
        n_before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset='sequence', keep='first')
        if len(combined_df) < n_before:
            print(f"WARNING: Dropped {n_before - len(combined_df)} duplicate(s) at combination step (should not happen with upstream dedup)")
        print(f"Sending {len(combined_df)} total sequences to lab")
        
        acquired_data_dict, invalid_sequences = acquire_sequences(
            sequences_df=combined_df,
            target_names=target_names,
            target_to_index_mapping=target_to_index_mapping,
            threshold=args.activity_threshold,
            zero_shot_fitness_location=args.zero_shot_fitness_predictions_location
        )
        
        valid_sequences = [seq for seq in combined_sequences if seq not in invalid_sequences]
        if len(valid_sequences) == 0:
            print("All acquired sequences were invalid. Resending the same batch to the lab.")
            acquired_data_dict, invalid_sequences = acquire_sequences(
                sequences_df=combined_df,
                target_names=target_names,
                target_to_index_mapping=target_to_index_mapping,
                threshold=args.activity_threshold,
                zero_shot_fitness_location=args.zero_shot_fitness_predictions_location
            )
            valid_sequences = [seq for seq in combined_sequences if seq not in invalid_sequences]
        
        new_data_df = combined_df[combined_df['sequence'].isin(valid_sequences)].copy()
        
        if not new_data_df.empty:
            new_data_df['mutant'] = new_data_df["sequence"].apply(
                lambda x: generate_mutation_string(starting_sequence, x)
            )
            for target, seq_values in acquired_data_dict.items():
                new_data_df[target] = new_data_df['sequence'].map(seq_values)
            add_sequence_data(connection, table_name, new_data_df)
        
        # --- Retrain the shared model ---
        if iteration_idx < num_acquisitions:
            model, target_processing = train_model(
                args=args,
                tablespace_location=tablespace_location,
                database_name=database_name,
                table_name=table_name,
                model_checkpoint_folder=model_checkpoint_folder,
                model_name='_'.join([model_name, 'round', str(iteration_idx)]),
                embeddings_location=embeddings_location
            )
        print(f"Completed iteration {iteration_idx}/{num_acquisitions}")

    # Return all acquired sequences sorted by main target
    train_data_df, target_processing = get_train_database(args, tablespace_location, database_name, table_name)
    train_data_df = train_data_df.sort_values(by=phenotype_name, ascending=False).reset_index()
    print(f"Total acquired sequences: {len(train_data_df)}")
    
    connection.close()
    return train_data_df

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
    parser.add_argument('--target_to_index_mapping', type=str, help='Path to config file mapping target name to lab target index')
    #Common to all agents - Bayesian optimization params
    parser.add_argument('--sampling_mode', default="conditional_sampling", type=str, help='Approach to select sequences to score at each round [conditional_sampling|all_combinations|sample_at_random]')
    parser.add_argument('--num_acquisitions', type=int, help='Total number of data acquisitions (ie., number of Bayesian Optimization rounds)')
    parser.add_argument('--acquisition_batch', type=int, default=3, help='Number of datapoints to acquire at each round (per agent)')
    parser.add_argument('--num_samples_per_iteration', type=int, help='If not exhaustively scoring all remaining sequences at each round, this is the number of sequences to sample and score')
    parser.add_argument('--acquisition_methods', default="ucb", type=str, help='Comma-separated list of acquisition functions for multi-agent BO (e.g., spec1,spec2,spec3)')
    parser.add_argument('--activity_threshold', default=0.001, type=float, help='Activity threshold value beyond which sequence is deemed active')
    #Agent-specific
    parser.add_argument('--target_config_location', type=str, help='Path to target config file')
    parser.add_argument('--model_name_suffix', type=str, help='Model name suffix')
    #Resume params
    parser.add_argument('--seed_sequences', default=None, type=str, help='Path to CSV with pre-measured sequences to seed the DB (skips parent init)')
    parser.add_argument('--starting_round', default=0, type=int, help='BO round to resume from (0=normal start, >0 skips init)')

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
    print(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    args = parse_arguments()
    random.seed(42)
    sequence_segments = convert_fragment_df_to_dict(pd.read_csv(args.sequence_segments_location, index_col=0, low_memory=False))
    print("Initial segments loaded:", sequence_segments)
    args.save_model_checkpoint = not args.do_not_save_model_checkpoint
    args.frozen_embedding_parameters = not args.fine_tune_model_embedding_parameters
    if not os.path.exists(args.data_location+os.sep+'data'): os.makedirs(args.data_location+os.sep+'data')

    acquisition_methods = [m.strip() for m in args.acquisition_methods.split(",")]
    print(f"Running with {len(acquisition_methods)} agent(s): {acquisition_methods}")

    print("\nDEBUG sequence_segments structure:")
    print(f"Number of segment positions: {len(sequence_segments)}")
    for key in sorted(list(sequence_segments.keys()))[:5]:
        print(f"Key: {key}, Number of segments: {len(sequence_segments[key])}")
        print(f"First segment length: {len(sequence_segments[key][0])}")
        print(f"First segment: {sequence_segments[key][0][:50]}...")

    total_length = sum(len(sequence_segments[key][0]) for key in sequence_segments)
    print(f"Total expected sequence length from all segments: {total_length}")
    
    results = optimize_starting_sequence(
        starting_sequence=args.starting_sequence, 
        args=args,
        sequence_segments=sequence_segments,
        num_acquisitions=args.num_acquisitions, 
        acquisition_batch=args.acquisition_batch, 
        num_samples_per_iteration=args.num_samples_per_iteration,
        sampling_mode=args.sampling_mode, 
        uncertainty_method=args.uncertainty_method, 
        acquisition_methods=acquisition_methods,
        selection_method=args.selection_method,
        model_name=args.model_name_suffix,
        target_config_location=args.target_config_location,
        target_to_index_mapping=args.target_to_index_mapping,
        model_checkpoint_folder=args.model_checkpoint_folder,
        embeddings_location=args.embeddings_location,
        num_MC_dropout_samples=args.num_MC_dropout_samples,
        tablespace_location=args.tablespace_location,
        database_name=args.database_name,
        table_name=args.table_name,
        ESM_location=args.ESM_location,
        seed_sequences=args.seed_sequences,
        starting_round=args.starting_round
    )

    print(results)
    if not os.path.exists(args.data_location+os.sep+'BO_results'): os.makedirs(args.data_location+os.sep+'BO_results')
    results.to_csv(args.data_location+os.sep+'BO_results'+os.sep+args.model_name_suffix+'.csv')
