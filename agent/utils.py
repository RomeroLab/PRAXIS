import os,json,sys
import numpy as np
import pandas as pd
import sqlite3
import psycopg2
from psycopg2 import sql
from datasets import Dataset
from collections import defaultdict
import torch
import tqdm
from Bio import SeqIO
import re

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data.sampler import SequentialSampler
from transformers import DataCollatorForLanguageModeling
from esm.pretrained import load_model_and_alphabet

# Re-export ConvBertLayer onto transformers so proteinnpt imports (see _compat_proteinnpt).
import _compat_proteinnpt  # noqa: F401
from proteinnpt.utils.data_utils import preprocess_training_targets, collate_fn_protein_npt, slice_sequences
from proteinnpt.proteinnpt.data_processing import process_batch
from proteinnpt.utils.tranception.utils.scoring_utils import get_sequence_slices
from proteinnpt.utils.msa_utils import weighted_sample_MSA, align_new_sequences_to_msa

def connect_db(tablespace_location, database_name, db_type="sqlite3", username=None, password=None):
    if db_type=="sqlite3":
        try:
            # Connect to a SQLite database; create the file if it doesn't exist
            if not os.path.exists(tablespace_location): os.makedirs(tablespace_location)
            connection = sqlite3.connect(tablespace_location + os.sep + database_name)
            print("Connection to the database successful")
            return connection
        except sqlite3.Error as e:
            print(f"Error connecting to the database: {e}")
            return None
    elif db_type=="postgresql":
        try:
            connection = psycopg2.connect(
                host=os.environ.get("PRAXIS_DB_HOST", "localhost"),
                database=database_name,
                user=username or os.environ.get("PRAXIS_DB_USER"),
                password=password or os.environ.get("PRAXIS_DB_PASSWORD"),
                port=int(os.environ.get("PRAXIS_DB_PORT", 5432))
            )
            print("Connection to the database successful")
            return connection
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

def create_tablespace(connection, tablespace_name, location, db_type="sqlite3"):
    if not os.path.exists(location): os.makedirs(location)
    """Create a new PostgreSQL tablespace at the specified location."""
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE TABLESPACE {} LOCATION {}").format(
                sql.Identifier(tablespace_name),
                sql.Literal(location)
            )
        )
        connection.commit()

def create_table(connection, table_name, target_names, database_name=None, db_type="sqlite3"):
    if db_type=="sqlite3":
        cursor = connection.cursor()

        # Adjusting column definitions for SQLite
        columns = [
            "sequence_id INTEGER PRIMARY KEY AUTOINCREMENT", 
            "mutant TEXT",
            "sequence TEXT UNIQUE",
            "segmented_sequence TEXT",  # SQLite doesn't support array types
            "created_by TEXT",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        columns += [f"{name} FLOAT" for name in target_names]

        create_table_command = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"
        
        try:
            cursor.execute(create_table_command)
            connection.commit()
            print("Table created successfully")
        except Exception as error:
            print("Error while creating SQLite table", error)
        finally:
            cursor.close()

    elif db_type=="postgresql":
        # Establishing the connection
        conn = connect_db(database_name)
        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        # Dynamically building the columns part of the SQL statement
        columns = [
            sql.SQL("sequence_id SERIAL PRIMARY KEY"), 
            sql.SQL("mutant TEXT"), 
            sql.SQL("sequence TEXT"), 
            sql.SQL("segmented_sequence TEXT[]"),
            sql.SQL("created_by TEXT"),
            sql.SQL("created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")
        ]
        columns += [sql.SQL("{} FLOAT").format(sql.Identifier(name)) for name in target_names]

        # Building the complete SQL CREATE TABLE command
        create_table_command = sql.SQL("CREATE TABLE IF NOT EXISTS {table} (") + \
                            sql.SQL(", ").join(columns) + \
                            sql.SQL(");")
        
        try:
            # Executing the SQL command
            cursor.execute(create_table_command.format(table=sql.Identifier(table_name)))
            # Commit your changes in the database
            conn.commit()
            print("Table created successfully")
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while creating PostgreSQL table", error)
        finally:
            # Closing the connection
            if conn is not None:
                cursor.close()
                conn.close()

def create_tablespace(connection, tablespace_name, location):
    """Create a new PostgreSQL tablespace at the specified location."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE TABLESPACE {} LOCATION {}").format(
                    sql.Identifier(tablespace_name),
                    sql.Literal(location)
                )
            )
            connection.commit()
            print(f"Tablespace '{tablespace_name}' created successfully at {location}")
    except Exception as e:
        print(f"Failed to create tablespace: {e}")
        connection.rollback()

def fetch_data_for_training(connection, table_name, properties, db_type="sqlite3"):
    if db_type=="sqlite3":
        cursor = connection.cursor()
        columns = ['mutant', 'sequence', 'segmented_sequence'] + properties
        columns_sql = ", ".join(columns)  # simpler column handling

        query = f"SELECT {columns_sql} FROM {table_name}"

        try:
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
            return df
        except Exception as error:
            print("Failed to fetch data:", error)
        finally:
            cursor.close()

    elif db_type=="postgresql":
        # Create the column list for SQL query, assuming 'sequence' is always present
        columns = ['mutant', 'sequence', 'segmented_sequence'] + properties
        columns_sql = sql.SQL(", ").join(map(sql.Identifier, columns))

        # Prepare SQL query
        query = sql.SQL("SELECT {} FROM {}").format(
            columns_sql,
            sql.Identifier(table_name)
        )

        # Execute query and fetch data into a pandas DataFrame
        with connection.cursor() as cursor:
            cursor.execute(query)
            # Fetch the results and convert to DataFrame
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
    
    return df

def add_sequence_data(connection, table_name, df, db_type="sqlite3"):
    if db_type=="sqlite3":
        cursor = connection.cursor()

        properties = [col for col in df.columns if col != 'sequence_id']  # Exclude 'sequence_id' if auto-increment
        placeholders = ", ".join(["?"] * len(properties))
        columns_string = ", ".join(properties)
        update_set = ", ".join([f"{prop}=excluded.{prop}" for prop in properties])

        sql_query = f"""
            INSERT INTO {table_name} ({columns_string}, created_at)
            VALUES ({placeholders}, CURRENT_TIMESTAMP)
            ON CONFLICT(sequence) DO UPDATE SET
            {update_set}
        """

        try:
            for index, row in df.iterrows():
                # Prepare data tuple, converting lists or dicts to JSON strings if necessary
                data_tuple = tuple(
                    json.dumps(item) if isinstance(item, (list, dict)) else item
                    for item in (row[col] for col in properties)
                )
                #print("Data tuple to insert/update:", data_tuple)  # Debug print
                cursor.execute(sql_query, data_tuple)
            connection.commit()
            #for index, row in df.iterrows():
            #    data_tuple = tuple(row[col] for col in properties)
            #    cursor.execute(sql_query, data_tuple)
            #connection.commit()
        except Exception as error:
            print("Failed to add/update data:", error)
        finally:
            cursor.close()
    
    elif db_type=="postgresql":
        # Columns to exclude from updates (typically PK or columns that should not be updated)
        exclude_from_update = ['sequence']  # Assuming 'sequence' is the unique identifier

        # Prepare columns for SQL statement
        properties = [col for col in df.columns if col not in exclude_from_update]
        all_properties = properties + ['created_at']  # Assuming 'created_at' is managed within the DB as CURRENT_TIMESTAMP

        # SQL identifiers for columns
        columns = sql.SQL(", ").join([sql.Identifier(prop) for prop in properties])
        placeholders = sql.SQL(", ").join([sql.Placeholder(prop) for prop in properties])

        # Prepare the SQL for updating on conflict
        updates = sql.SQL(", ").join([
            sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(prop))
            for prop in properties if prop not in exclude_from_update
        ])

        # Constructing the full SQL command
        sql_query = sql.SQL("""
            INSERT INTO {table} ({columns}, created_at)
            VALUES ({placeholders}, CURRENT_TIMESTAMP)
            ON CONFLICT (sequence) DO UPDATE SET
            {updates}
        """).format(
            table=sql.Identifier(table_name),
            columns=columns,
            placeholders=placeholders,
            updates=updates
        )

        # Execute SQL for each row in the DataFrame
        try:
            for index, row in df.iterrows():
                data_tuple = tuple(row[col] for col in properties)
                cursor.execute(sql_query, data_tuple)
            connection.commit()
        except Exception as error:
            print("Failed to add/update data:", error)
        finally:
            cursor.close()

def setup_config_and_paths(args):
    # Model config
    if args.model_config_location is not None:
        with open(args.model_config_location, 'r') as file:
            args.main_config = json.load(file)
        args_setup_from_config=set([])
        for key in args.main_config:
            if (key not in args.__dict__) or (args.__dict__[key] is None):
                args.__dict__[key] = args.main_config[key]
                args_setup_from_config.add(key)
    else:
        print("Model config file not detected. Aborting")
        sys.exit(0)

    # File paths config
    for local_path in ['embedding_model_location','MSA_data_folder','MSA_weight_data_folder','path_to_hhfilter']:
        if getattr(args, local_path) and local_path in args_setup_from_config:
            setattr(args, local_path, args.data_location + os.sep + getattr(args, local_path))
    for directory in ['model_predictions','checkpoint', 'model_predictions' + os.sep + args.model_name_suffix, 'checkpoint' + os.sep + args.model_name_suffix]:
        if not os.path.exists(args.data_location + os.sep + directory):
            os.makedirs(args.data_location + os.sep + directory)
    args.output_scores_location = args.data_location + os.sep + 'model_predictions' + os.sep + args.model_name_suffix
    args.model_location = args.data_location + os.sep + 'checkpoint' + os.sep + args.model_name_suffix
    if not os.path.exists(args.model_checkpoint_folder): os.makedirs(args.model_checkpoint_folder)

    # Target config
    args.target_config=json.load(open(args.target_config_location))
    zero_shot_predictions_mapping={
            "MSA_Transformer_pred": "MSA_Transformer_ensemble",
            "ESM1v_pred": "ESM1v_ensemble",
            "ESM2_15B_pred": "ESM2_15B",
            "ESM2_3B_pred": "ESM2_3B",
            "ESM2_650M_pred": "ESM2_650M",
            "TranceptEVE_pred": "TranceptEVE_L",
            "Tranception_pred": "Tranception_L",
        }
    if args.model_type=="ProteinNPT": zero_shot_predictions_mapping["ProteinNPT"]=zero_shot_predictions_mapping[args.aa_embeddings+"_pred"]
    if args.augmentation=="zero_shot_fitness_predictions_auxiliary_labels": # Add auxiliary label to target_config
        assert args.zero_shot_fitness_predictions_location is not None, "Location of zero-shot fitness predictions to use as auxiliary labels not properly referenced"
        print("Using zero-shot fitness predictions as auxiliary labels")
        args.target_config["zero_shot_fitness_predictions"] = {
            "type": "continuous",
            "dim": 1,
            "var_name": zero_shot_predictions_mapping[args.model_type], #Select the relevant model for zero-shot fitness predictions
            "location": args.zero_shot_fitness_predictions_location,
            "in_NPT_loss": False,
            "main_target": False
        }
        args.augmentation_short="auxiliary"
    elif args.augmentation=="zero_shot_fitness_predictions_covariate": # Will use zero-shot fitness predictions as an additional model covariate
        assert args.zero_shot_fitness_predictions_location is not None, "Location of zero-shot fitness predictions to use as model covariate not properly referenced"
        print("Using zero-shot fitness predictions as covariate")
        args.augmentation_short="covariate"
        args.zero_shot_fitness_predictions_var_name = zero_shot_predictions_mapping[args.model_type]
    else:
        args.augmentation_short="none"
    
    for target in args.target_config:
        if "location" not in args.target_config[target].keys(): 
            args.target_config[target]["location"] = args.tablespace_location
        
    return args

def collapse_triplets(s):
    triplets = s.split(":")
    positions = {}
    for triplet in triplets:
        pos = triplet[1:-1]  # The position is the middle character
        aa1, aa2 = triplet[0], triplet[-1]  # The amino acids are the first and last characters
        if pos in positions:
            positions[pos] = positions[pos][:-1] + aa2
        else:
            positions[pos] = aa1 + pos + aa2
    s_new = ":".join([aa for aa in positions.values()])
    return s_new

def convert_fragment_df_to_dict(dataframe, mode="segments_at_index_position"):
    """
    If segments_at_index_position mode, returns a dict in which keys index over segments and the corresponding values list all eligible fragments at that index position.
    If parent_segments mode, returns a dict in which keys index over parents and the corresponding values list the fragments that make up this parent.
    """
    if mode=="segments_at_index_position":
        return {
            col: dataframe[col].dropna().tolist() for col in dataframe.columns
        }
    elif mode=="parent_segments":
        return {
            index: [row[f'f{i}'] for i in range(len(dataframe.columns)) if pd.notna(row[f'f{i}'])]
            for index, row in dataframe.iterrows()
        }
    else:
        print("Fragment conversion mode not yet implemented")
        sys.exit(0)

def get_train_database(args, tablespace_location, database_name, table_name):
    target_names = list(args.target_config.keys())
    connection = connect_db(
        tablespace_location=tablespace_location,
        database_name=database_name
    )
    df = fetch_data_for_training(connection, table_name, target_names) 
    
    raw_targets = {target_name: df[target_name] for target_name in target_names}

    # Epsilon guard: if a continuous target has zero variance (e.g. all-zero "dead enzyme" round),
    # preprocess_training_targets would divide by std=0 and produce NaN. We prevent this by
    # adding a tiny deterministic spread, matching the standard practice of BatchNorm/LayerNorm
    # (eps parameter). The DB is left untouched; only the in-memory copy used for normalization
    # is perturbed. The model correctly receives near-zero uniform signal for this target.
    for target_name in target_names:
        if args.target_config.get(target_name, {}).get("type") == "continuous":
            arr = raw_targets[target_name].to_numpy(dtype=float)
            if np.nanstd(arr) == 0:
                print(f"Warning: target '{target_name}' has zero variance. Applying epsilon guard for normalization.")
                jitter = np.linspace(0, 1e-8, len(arr))
                raw_targets[target_name] = pd.Series(arr + jitter, index=raw_targets[target_name].index)

    preprocessed_targets, target_processing = preprocess_training_targets(raw_targets, args.target_config)
    for target_name in target_names:
        df[target_name] = preprocessed_targets[target_name]
    #train_data = Dataset.from_pandas(df)
    return df, target_processing

def data_cleanup_for_model_input(df, del_vars=True):
    df['sequence'] = df['sequence'].apply(lambda x: x.replace("*",""))
    df['mutant_mutated_seq_pairs'] = list(zip(df['mutant'],df['sequence']))
    if del_vars:
        del df['segmented_sequence']
        del df['sequence']
        del df['mutant']
    return df

def score_mutated_sequences(model, test_data_df, training_data_df, target_processing, return_uncertainty=False, uncertainty_method=None, num_MC_dropout_samples=10, MSA_sequences = None, MSA_weights = None, MSA_start_position = None, MSA_end_position = None, selected_indices_seed=0, verbose=True):
    """
    Returns predictions and corresponding uncertainties (optional) for all input mutated_sequences.
    """
    model.eval()
    model.cuda()
    # Ensure training targets are numeric floats to avoid numpy.object_ arrays downstream
    try:
        for target_name in getattr(model, 'target_names', []) or []:
            if target_name in training_data_df.columns:
                training_data_df[target_name] = pd.to_numeric(training_data_df[target_name], errors='coerce').astype(float)
    except Exception:
        pass
    test_dataset = Dataset.from_pandas(test_data_df)
    train_dataset = Dataset.from_pandas(training_data_df)
    model_output = defaultdict(list)
    targets_to_predict = model.target_names #[target for target in model.target_names_input if target != "zero_shot_fitness_predictions"]
    with torch.no_grad():
        eval_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset, 
                            batch_size=model.args.eval_num_sequences_to_score_per_batch_per_gpu, 
                            shuffle=False,
                            num_workers=model.args.num_data_loaders_workers,
                            pin_memory=True,
                            collate_fn=collate_fn_protein_npt
                        )
        eval_iterator = iter(eval_loader)
        iterable = tqdm.tqdm(eval_iterator, disable=not verbose, desc="Scoring sequences") if verbose else eval_loader
        for batch in iterable:
            processed_batch = process_batch(
                        batch = batch,
                        model = model,
                        alphabet = model.alphabet, 
                        args = model.args, 
                        MSA_sequences = MSA_sequences,
                        MSA_weights = MSA_weights,
                        MSA_start_position = MSA_start_position, 
                        MSA_end_position = MSA_end_position,
                        target_processing = target_processing,
                        training_sequences = train_dataset,
                        proba_target_mask = 1.0, 
                        proba_aa_mask = 0.0,
                        eval_mode = True,
                        device=model.device,
                        selected_indices_seed=selected_indices_seed,
                        indel_mode=model.args.indel_mode
                    )
            if model.args.augmentation=="zero_shot_fitness_predictions_covariate":
                zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                del processed_batch['target_labels']['zero_shot_fitness_predictions']
            else:
                zero_shot_fitness_predictions = None
            
            if not return_uncertainty:
                output = model.forward(
                    tokens=processed_batch['masked_tokens'],
                    targets=processed_batch['masked_targets'],
                    zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                    sequence_embeddings=processed_batch['sequence_embeddings']
                )
                num_of_mutated_seqs_to_score = processed_batch['num_of_mutated_seqs_to_score'] if model.model_type=="ProteinNPT" else len(processed_batch['mutant_mutated_seq_pairs'])
                for target_name in targets_to_predict:
                    model_output["predictions_avg_"+target_name] += list(output["target_predictions"][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
                assert output['logits_protein_sequence'].shape[0]==1, "Found multiple batches in input"
                model_output["logits_protein_sequence"] += list(output['logits_protein_sequence'][0][:num_of_mutated_seqs_to_score].cpu().numpy()) # output['logits_protein_sequence'] of shape (1,num_sequences,L,vocab)
            else:
                output = model.forward_with_uncertainty(
                    tokens=processed_batch['masked_tokens'],
                    targets=processed_batch['masked_targets'],
                    zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                    sequence_embeddings=processed_batch['sequence_embeddings'],
                    num_MC_dropout_samples=num_MC_dropout_samples
                )
                num_of_mutated_seqs_to_score = processed_batch['num_of_mutated_seqs_to_score'] if model.model_type=="ProteinNPT" else len(processed_batch['mutant_mutated_seq_pairs'])
                for target_name in targets_to_predict:
                    model_output["predictions_avg_"+target_name] += list(output[target_name]["predictions_avg"][:num_of_mutated_seqs_to_score].cpu().numpy())
                    model_output["uncertainty_"+target_name] += list(output[target_name]["uncertainty"][:num_of_mutated_seqs_to_score].cpu().numpy())
            model_output["mutant_mutated_seq_pairs"] += processed_batch["mutant_mutated_seq_pairs"][:num_of_mutated_seqs_to_score]

    model_output["mutant"], model_output["sequence"] = zip(*model_output["mutant_mutated_seq_pairs"])
    model_output = pd.DataFrame(model_output)
    return model_output

def get_ESM_dataloader(df, batch_size):
    dataset_dict = {}
    dataset_dict['mutant_mutated_seq_pairs'] = list(zip(list(df['mutant']),list(df['mutated_sequence'])))
    dataset = Dataset.from_dict(dataset_dict)
    dataloader = DataLoader(
                    dataset=dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=0, 
                    pin_memory=True,
                    collate_fn=collate_fn_protein_npt
                )
    return dataloader

def get_Tranception_dataloader(df, batch_size, model, target_seq, indel_mode):
    sliced_df = get_sequence_slices(df, 
        target_seq=target_seq, 
        model_context_len = model.config.n_ctx - 2, 
        indel_mode=indel_mode, 
        scoring_window="optimal"
    )
    mutant_index=0
    ds = Dataset.from_pandas(sliced_df)
    ds.set_transform(model.encode_batch)
    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=model.config.tokenizer,
                        mlm=False)
    sampler = SequentialSampler(ds)
    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=data_collator, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=False
    )
    return dataloader

def fast_MSA_mode_setup(model, seqs_to_score, MSA_sequences, MSA_weights, num_MSA_sequences, path_to_clustalomega):
    model.MSA_sample_sequences = weighted_sample_MSA(
                MSA_all_sequences=MSA_sequences, 
                MSA_non_ref_sequences_weights=MSA_weights, 
                number_sampled_MSA_sequences=num_MSA_sequences
            )
    fast_MSA_short_names_mapping = {}
    fast_MSA_short_names = []
    print("There are {} sequences to score".format(len(seqs_to_score)))
    for seq_index,seq in enumerate(list(seqs_to_score)):
        short_name = 'mutant_to_score_'+str(seq_index)
        fast_MSA_short_names_mapping[seq] = short_name
        fast_MSA_short_names.append(short_name)
    fast_MSA_aligned_sequences = align_new_sequences_to_msa(model.MSA_sample_sequences, list(seqs_to_score), fast_MSA_short_names, clustalomega_path=path_to_clustalomega)
    model.MSA_sample_sequences = [tup for tup in fast_MSA_aligned_sequences if tup[0] not in set(fast_MSA_short_names)]
    return model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping

def process_MSA_Transformer_batch(batch, model, alphabet, max_positions, MSA_start_position, MSA_end_position, MSA_weights, MSA_sequences, num_MSA_sequences, eval_mode, start_idx=1, long_sequences_slicing_method="left", indel_mode=False, fast_MSA_mode=False, clustalomega_path=None, batch_target_labels=None, batch_masked_targets=None, target_names=None, num_extra_tokens=1):
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) #Selects the first mutant-seq pair, and the sequence is 1-indexed in that pair
    raw_batch_size = len(batch['mutant_mutated_seq_pairs']) # Effective number of mutated sequences to be scored (could be lower than num_MSA_sequences in practice, eg., if we're at the end of the train_iterator)
    if (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)):
        MSA_start_index = MSA_start_position - 1
        MSA_end_index = MSA_end_position
        batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in batch['mutant_mutated_seq_pairs']]
        # Recompute sequence length (has potentially been chopped off above)
        raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
    #Sample MSA sequences based on sequence weights        
    assert MSA_weights is not None, "Trying to add MSA_sequences to scoring batch but no weights are provided"
    if model.MSA_sample_sequences is None:
        model.MSA_sample_sequences = weighted_sample_MSA(
            MSA_all_sequences=MSA_sequences, 
            MSA_non_ref_sequences_weights=MSA_weights, 
            number_sampled_MSA_sequences=num_MSA_sequences
        )
    # Concatenate MSA sequences with labelled assay sequences
    batch['mutant_mutated_seq_pairs'] += model.MSA_sample_sequences
    if max_positions is not None and raw_sequence_length + 1 > max_positions: # Adding one for the BOS token
        if long_sequences_slicing_method=="center":
            #print("Center slicing method not adapted to PNPT embeddings as sequences would not be aligned in the same system anymore. Defaulting to 'left' mode.")
            long_sequences_slicing_method="left"
        batch['mutant_mutated_seq_pairs'], batch_target_labels, batch_masked_targets, batch_scoring_optimal_window = slice_sequences(
            list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
            max_positions=max_positions,
            method=long_sequences_slicing_method,
            rolling_overlap=max_positions//4,
            eval_mode=eval_mode,
            batch_target_labels=batch_target_labels,
            batch_masked_targets=batch_masked_targets, 
            start_idx=start_idx,
            target_names=target_names,
            num_extra_tokens=num_extra_tokens
        )
    #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
    num_sequences = raw_batch_size + num_MSA_sequences
    assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, "Unexpected number of sequences"
    sequences_to_score = batch['mutant_mutated_seq_pairs'][:raw_batch_size]
    MSA_sequences = batch['mutant_mutated_seq_pairs'][raw_batch_size:]
    
    if fast_MSA_mode:
        mutants_to_score, seqs_to_score = zip(*sequences_to_score)
        model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping = fast_MSA_mode_setup(model, seqs_to_score, MSA_sequences, MSA_weights, num_MSA_sequences, clustalomega_path)
    else:
        fast_MSA_aligned_sequences = None
        fast_MSA_short_names_mapping = None
    
    if indel_mode and not fast_MSA_mode: 
        mutant_to_score, new_sequence = sequences_to_score[0]
        MSA_sequences = align_new_sequences_to_msa(MSA_sequences, [new_sequence], ["mutant_to_score"], clustalomega_path=clustalomega_path)
        sequences_to_score = [(mutant_to_score,tup[1]) for tup in MSA_sequences if tup[0] == "mutant_to_score"] #Have to resort to this otherwise "mutant_to_score" is truncated
        MSA_sequences = [tup for tup in MSA_sequences if tup[0] != "mutant_to_score"]
    elif indel_mode and fast_MSA_mode:
        seq_scoring = []
        for sequence in sequences_to_score:
            seq_short_name = fast_MSA_short_names_mapping[sequence[0]]
            score_tuple = [(sequence[0],tup[1]) for tup in fast_MSA_aligned_sequences if tup[0] == seq_short_name]
            assert len(score_tuple)==1
            seq_scoring.append(score_tuple[0])
        sequences_to_score = seq_scoring
    
    if fast_MSA_mode:
        batch['mutant_mutated_seq_pairs'] = [sequences_to_score + MSA_sequences]
    else:
        batch['mutant_mutated_seq_pairs'] = [ [sequence] + MSA_sequences for sequence in sequences_to_score]
    
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    if batch_target_labels is not None: processed_batch['target_labels'] = batch_target_labels
    if batch_masked_targets is not None: processed_batch['masked_targets'] = batch_masked_targets
    return processed_batch

def process_MSA_Transformer_batch_multiple_MSAs(batch, model, alphabet, MSA_folder, num_MSA_sequences, batch_target_labels=None, batch_masked_targets=None):
    """
    There is a unique MSA used for each input sequence. Does not have all details from MSA processing "single MSA" yet.
    We also do not worry about sequence weights to keep things computationally tractable.
    Assume a batch size of 1 --> mutant[0]
    """
    mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
    MSA_sequences = [
            (record.description, str(record.seq)) for record in SeqIO.parse(MSA_folder + os.sep + mutant[0] + ".a3m", "fasta")
    ]
    model.MSA_sample_sequences = weighted_sample_MSA(
            MSA_all_sequences=MSA_sequences, 
            MSA_non_ref_sequences_weights=None, 
            number_sampled_MSA_sequences=num_MSA_sequences
    )
    batch['mutant_mutated_seq_pairs'] = [ [batch['mutant_mutated_seq_pairs']] + model.MSA_sample_sequences ]
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    if batch_target_labels is not None: processed_batch['target_labels'] = batch_target_labels
    if batch_masked_targets is not None: processed_batch['masked_targets'] = batch_masked_targets
    return processed_batch

def process_ESM_batch(batch, model, alphabet, max_positions, long_sequences_slicing_method, eval_mode, start_idx=1, batch_target_labels=None, target_names=None, num_extra_tokens=1):
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) #Selects the first mutant-seq pair, and the sequence is 1-indexed in that pair
    if max_positions is not None and raw_sequence_length + 1 > max_positions: # Adding one for the BOS token
        batch['mutant_mutated_seq_pairs'], batch_target_labels, _ = slice_sequences(
                list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
                max_positions=max_positions,
                method=long_sequences_slicing_method,
                rolling_overlap=max_positions//4,
                eval_mode=eval_mode,
                batch_target_labels=batch_target_labels, 
                start_idx=start_idx,
                target_names=target_names,
                num_extra_tokens=num_extra_tokens
            )
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    return processed_batch

def process_Tranception_batch(batch, model):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    return batch

def get_embeddings_MSA_Transformer(model, processed_batch, alphabet_size, return_pseudo_ll=True, fast_MSA_mode=False):
    tokens = processed_batch['input_tokens']
    assert tokens.ndim == 3, "Finding dimension of tokens to be: {}".format(tokens.ndim)
    num_MSAs_in_batch, num_sequences_in_alignments, seqlen = tokens.size()
    batch_size = num_MSAs_in_batch
    output = model(tokens, repr_layers=[12])
    embeddings = output["representations"][12][:] # B, N, L, D
    if fast_MSA_mode: # A single batch with multiple sequences to score in it to speed things up
        embeddings = embeddings[:,:batch_size,:,:] # 1, N, L, D 
        embeddings = [embeddings[:,i,:,:] for i in range(batch_size)]
        if return_pseudo_ll:
            logits = output["logits"][:,:batch_size].view(batch_size,-1,alphabet_size)
            tokens = tokens[:,:batch_size].view(batch_size,-1)
            pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(batch_size, seqlen)
            pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu()
            pseudo_ll = [psll.item() for psll in pseudo_ll]
        else:
            pseudo_ll = None
        if 'mutant_mutated_seq_pairs' in processed_batch: processed_batch['mutant_mutated_seq_pairs'] = processed_batch['mutant_mutated_seq_pairs'][0][:batch_size]
    else:
        embeddings = embeddings[:,0,:,:] # In each MSA batch the first sequence is what we care about. The other MSA sequences were present just to compute embeddings and logits
        if return_pseudo_ll:
            logits = output["logits"][:,0] # Filtering logits of points to score
            tokens = tokens[:,0] # Filtering tokens of points to score
            pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(num_MSAs_in_batch, seqlen)
            pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu() #Average across sequence length
            pseudo_ll = [psll.item() for psll in pseudo_ll]
        else:
            pseudo_ll = None
        if 'mutant_mutated_seq_pairs' in processed_batch: processed_batch['mutant_mutated_seq_pairs'] = [seq[0] for seq in processed_batch['mutant_mutated_seq_pairs']] # Remove MSA sequences from batch by selecting the first sequence in each MSA input sets
    return embeddings, pseudo_ll, processed_batch

def get_embeddings_ESM(model, model_type, processed_batch, alphabet_size, return_pseudo_ll=True):
    tokens = processed_batch['input_tokens']
    assert tokens.ndim == 2, "Finding dimension of tokens to be: {}".format(tokens.ndim)
    batch_size, seqlen = tokens.size()
    if model_type=="ESM1v":
        last_layer_index = 33
    elif model_type=="ESM2_15B":
        last_layer_index = 48
    elif model_type=="ESM2_3B":
        last_layer_index = 36
    elif model_type=="ESM2_650M":
        last_layer_index = 33
    output = model(tokens, repr_layers=[last_layer_index])
    embeddings = output["representations"][last_layer_index][:] # N, L, D
    if return_pseudo_ll:
        logits = output["logits"]
        pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.view(-1, alphabet_size), tokens.view(-1)).view(batch_size, seqlen)
        pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu()
        pseudo_ll = [psll.item() for psll in pseudo_ll]
    else:
        pseudo_ll = None
    return embeddings, pseudo_ll

def get_embeddings_Tranception(model, processed_batch, return_pseudo_ll=True):
    output = model(**processed_batch, return_dict=True, output_hidden_states=True)
    embeddings = output.hidden_states[-1] # Extract embeddings from last layer
    if return_pseudo_ll:
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = processed_batch['labels'][..., 1:].contiguous()
        pseudo_ll = - CrossEntropyLoss(reduction='none')(input=shift_logits.view(-1, shift_logits.size(-1)), target=shift_labels.view(-1)).view(shift_logits.shape[0],shift_logits.shape[1])
        pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu() # Shape (B,)
        pseudo_ll = [psll.item() for psll in pseudo_ll]
    else:
        pseudo_ll = None
    return embeddings, pseudo_ll

def add_back_embedding(model):
    """
    Add back aa_embedding if previously deleted (eg., to save checkpoint)
    """
    if not hasattr(model, 'aa_embedding'):
        if model.args.aa_embeddings.startswith("ESM"):
            model.aa_embedding = load_model_and_alphabet(model.args.embedding_model_location)[0]
        else:
            print("Model type not implemented yet")
            sys.exit(0)

def tokenize_sequence(sequence):
    # This function uses a regular expression to split the sequence into tokens,
    # treating '<mask>' as a single token and other characters individually
    return re.findall(r'<mask>|.', sequence)

def generate_mutation_string(starting_sequence, mutated_sequence):
    # Tokenize both sequences
    wt_tokens = tokenize_sequence(starting_sequence)
    mut_tokens = tokenize_sequence(mutated_sequence)

    # Generate mutation strings
    mutations = [
        wt + str(i+1) + mut
        for i, (wt, mut) in enumerate(zip(wt_tokens, mut_tokens))
        if wt != mut
    ]

    return ':'.join(mutations)

def get_segmented_sequence(row):
    if isinstance(row['segmented_sequence'], str):
        try:
            # Try to parse as JSON
            segmented_sequence_list = json.loads(row['segmented_sequence'])
        except json.JSONDecodeError:
            # If JSON decoding fails, treat as comma-separated values
            segmented_sequence_list = row['segmented_sequence'].split(',')
    else:
        # Handle the case where the data is already a list
        segmented_sequence_list = row['segmented_sequence']
    return segmented_sequence_list

def get_best_chunk(model, sequence_segments, masked_segment_index, log_probas):
    # Assuming 'log_probas' is correctly sliced to get probabilities for the masked segment
    # and 'model' has a way to encode segments into indices
    chunks = sequence_segments[f'f{masked_segment_index}']
    chunks = [chunk.replace("*","") for chunk in chunks]
    best_chunk = max(chunks, key=lambda chunk: np.mean(
        log_probas[:, model.alphabet.encode(chunk)].mean(dim=1).numpy()
    ))
    return best_chunk

def sample_chunk(model, sequence_segments, masked_segment_index, log_probas):
    # Get the list of possible chunks for the masked segment
    chunks = sequence_segments[f'f{masked_segment_index}']
    chunks = [chunk.replace("*","") for chunk in chunks]
    
    # Calculate probabilities from log probabilities
    original_chunk_len, _ = log_probas.shape
    # First, decode the chunks into model indices and get log probabilities for each chunk
    chunk_log_probs = [log_probas[torch.arange(min(original_chunk_len,len(chunk))), model.alphabet.encode(chunk[:min(original_chunk_len,len(chunk))])] for chunk in chunks]
    # Convert log probabilities to probabilities and compute the mean probability for each chunk
    chunk_probs = [torch.exp(log_prob).mean().item() for log_prob in chunk_log_probs]
    # Normalize the probabilities to sum to 1
    total_prob = sum(chunk_probs)
    normalized_probs = [prob / total_prob for prob in chunk_probs]
    
    # Randomly sample a chunk based on the normalized probabilities
    sampled_chunk = np.random.choice(chunks, p=normalized_probs)
    
    return sampled_chunk

def add_end_token_to_generated_sequence(generated_sequence, generated_segmented_sequence):
    if not generated_sequence.endswith('*'):
        generated_sequence += '*'
        generated_segmented_sequence[-1] = generated_segmented_sequence[-1] + '*'
    return generated_sequence, generated_segmented_sequence

def fetch_previously_acquired_sequences(db_path, table_name):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    try:
        # SQL query to fetch all distinct sequences from the table
        query = f"SELECT DISTINCT sequence FROM {table_name}"
        cursor.execute(query)
        
        # Fetch all results; each row will contain one sequence
        sequences = cursor.fetchall()
        
        # Create a list to hold the sequences
        distinct_sequences = [seq[0] for seq in sequences] #fetchall returns a list of tuples. Each tuple here contains only a single element (sequence) hence why we use seq[0]
        return distinct_sequences
            
    except Exception as error:
        print("Failed to fetch sequences:", error)
        return []  # Return an empty list in case of an error
    
    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()