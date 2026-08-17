import os
import sys
from pathlib import Path
# Ensure project root is on sys.path for absolute imports (utils, self_driving)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import random
import argparse
import pandas as pd
from typing import List, Set, Tuple, Optional
import numpy as np
import torch
import math

from utils import connect_db, create_table, add_sequence_data, get_train_database, data_cleanup_for_model_input
from utils import add_back_embedding, generate_mutation_string, score_mutated_sequences
from utils import setup_config_and_paths
from self_driving import train_model, select_mutants_to_acquire, get_main_target_from_config, get_zero_shot_fitness_predictions
from insilico_analysis.onehot_surrogate import OneHotMLPSurrogate
from insilico_analysis.dataset_oracle.synevo import Oracle, Frontier

VALID_PROTEINS = ('gfp', 'gb1', 'pab1', 'ube4')


def build_initial_seed(frontier, seed_size: int, max_attempts: int = 20) -> List[str]:
    """Sample seed_size random sequences from the dataset as the initial labelled set.

    If the sampled seed has no reachable dataset neighbors (which stops the loop
    immediately at an empty frontier), re-sample another random seed_size, up to
    max_attempts.
    """
    chosen = random.sample(frontier.all, seed_size)
    for attempt in range(1, max_attempts + 1):
        if frontier.frontier(set(chosen)):
            break
        print(f"Initial seed (attempt {attempt}) has no reachable neighbors; re-sampling {seed_size}.")
        chosen = random.sample(frontier.all, seed_size)
    else:
        print(f"WARNING: no seed with reachable neighbors after {max_attempts} attempts; using last sample.")
    return [s + "*" for s in chosen]


def ensure_local_target_config(local_dir: str, protein: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    cfg_path = os.path.join(local_dir, f"{protein}_target_config.json")
    if not os.path.exists(cfg_path):
        cfg = {
            "fitness": {
                "type": "continuous",
                "dim": 1,
                "var_name": "functional_score",
                "in_NPT_loss": True,
                "main_target": True
            }
        }
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
    return cfg_path


def _tokenize_sequence(sequence: str) -> List[str]:
    # Keep consistent with utils.tokenize_sequence behavior for '<mask>' handling
    import re
    return re.findall(r'<mask>|.', sequence)


def _sample_aa_from_logits(logits_LV: np.ndarray, position_index: int, alphabet) -> str:
    # logits_LV: array of shape (L, vocab)
    # Convert to probabilities at the masked position
    import torch as _torch
    pos_logits = _torch.tensor(logits_LV[position_index])
    probs = _torch.softmax(pos_logits, dim=-1).cpu().numpy()
    # Build AA vocabulary indices from alphabet
    # Filter to standard amino acids only
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_token_indices = [alphabet.get_idx(aa) if hasattr(alphabet, 'get_idx') else alphabet.tok_to_idx.get(aa, None) for aa in aa_list]
    # Fallback for any None indices
    valid = [(aa, idx) for aa, idx in zip(aa_list, aa_token_indices) if idx is not None and idx < probs.shape[-1]]
    if not valid:
        # Default to alanine
        return 'A'
    aa_probs = np.array([probs[idx] for _, idx in valid], dtype=float)
    if aa_probs.sum() <= 0 or not np.isfinite(aa_probs).all():
        # Numerical fallback
        aa_probs = np.ones_like(aa_probs, dtype=float)
    aa_probs = aa_probs / aa_probs.sum()
    chosen_idx = np.random.choice(len(valid), p=aa_probs)
    return valid[chosen_idx][0]


def conditional_sample_one_step_candidates(
    model,
    args,
    train_data_df: pd.DataFrame,
    phenotype_name: str,
    frontier,
    labeled_set: Set[str],
    starting_sequence: str,
    num_samples: int,
    target_processing,
    conditional_sampling_batch_size: int = 30,
    max_positions_per_seq: Optional[int] = None,
) -> List[str]:
    """
    Mirror self_driving.py's conditional sampling pattern for non-fragmented sequences:
    - Select top-quartile training sequences (label boosting to max phenotype)
    - For a random subset, generate a random permutation of integer positions and iterate through them; at each iteration mask that position and sample the AA from model logits
    - Keep only 1-step sequences that exist in the dataset and are not yet labeled
    Returns sequences with '*' suffix.
    """
    # Compute top quartile set
    phenotype_max_value = train_data_df[phenotype_name].max()
    top_quantile_threshold = train_data_df[phenotype_name].quantile(0.75)
    filtered_data = train_data_df[train_data_df[phenotype_name] >= top_quantile_threshold]

    if len(filtered_data) > conditional_sampling_batch_size:
        conditional_sampling_set = filtered_data.sample(conditional_sampling_batch_size, random_state=42).copy().reset_index(drop=True)
    elif conditional_sampling_batch_size >= len(filtered_data) and len(train_data_df) > conditional_sampling_batch_size:
        conditional_sampling_set = filtered_data.copy().reset_index(drop=True)
        conditional_sampling_batch_size = len(filtered_data)
    else:
        conditional_sampling_set = train_data_df.copy().reset_index(drop=True)
        conditional_sampling_batch_size = len(train_data_df)

    # Label boosting (for conditioning consistency with self_driving.py)
    if phenotype_name in conditional_sampling_set.columns:
        conditional_sampling_set[phenotype_name] = phenotype_max_value

    # Prepare random position orders per sequence
    # Assume fixed length protein; truncate to max_positions_per_seq for compute
    # Use current sequence from the DB row
    position_orders: List[List[int]] = []
    base_sequences: List[str] = []
    for _, row in conditional_sampling_set.iterrows():
        base_seq = str(row['sequence']).replace('*', '')
        if not base_seq or base_seq == 'nan':
            continue
        L = len(base_seq)
        order = list(range(L))
        random.shuffle(order)
        if max_positions_per_seq is not None:
            order = order[:min(max_positions_per_seq, L)]
        position_orders.append(order)
        base_sequences.append(base_seq)

    if not base_sequences:
        return []

    generated_sequences: List[str] = []

    # Iterate across the masked position index like self_driving.py
    # For each iteration index k, mask position position_orders[row_index][k] for each sequence
    max_iters = max(len(o) for o in position_orders)

    # Use boosted training data for conditioning context
    train_df_for_scoring = data_cleanup_for_model_input(conditional_sampling_set.copy(), del_vars=False)
    # Ensure target columns are strictly float dtype to avoid numpy.object_
    for target_name in getattr(model, 'target_names', []):
        if target_name in train_df_for_scoring.columns:
            train_df_for_scoring[target_name] = pd.to_numeric(train_df_for_scoring[target_name], errors='coerce').astype(float).fillna(0.0)

    for k in range(max_iters):
        masked_sequences = []
        masked_meta: List[Tuple[int, int, str]] = []  # (row_idx, pos_idx, base_seq)
        for row_index, base_seq in enumerate(base_sequences):
            # If this row has fewer remaining positions, skip for this iteration
            if k >= len(position_orders[row_index]):
                continue
            pos_idx = position_orders[row_index][k]
            tokens = _tokenize_sequence(base_seq)
            tokens[pos_idx] = '<mask>'
            masked_seq = ''.join(tokens)
            masked_sequences.append(masked_seq)
            masked_meta.append((row_index, pos_idx, base_seq))

        if not masked_sequences:
            continue

        test_data_df = pd.DataFrame({
            'sequence': masked_sequences,
            'segmented_sequence': [[s] for s in masked_sequences]
        })
        test_data_df['mutant'] = test_data_df['sequence'].apply(lambda x: generate_mutation_string(starting_sequence, x))

        test_df_for_scoring = data_cleanup_for_model_input(test_data_df.copy())

        add_back_embedding(model)
        scored = score_mutated_sequences(
            model=model,
            test_data_df=test_df_for_scoring,
            training_data_df=train_df_for_scoring,
            target_processing=target_processing,
            return_uncertainty=False,
            verbose=False
        )

        logits_list = scored.get("logits_protein_sequence", [])
        for (row_index, pos_idx, base_seq), logits_LV in zip(masked_meta, logits_list):
            aa = _sample_aa_from_logits(np.array(logits_LV), pos_idx, model.alphabet)
            if aa == base_seq[pos_idx]:
                continue
            new_seq = base_seq[:pos_idx] + aa + base_seq[pos_idx + 1:]
            if new_seq in frontier._set and new_seq not in labeled_set:
                generated_sequences.append(new_seq + "*")
                if len(generated_sequences) >= num_samples:
                    # Deduplicate and return
                    return list(dict.fromkeys(generated_sequences))

    # Deduplicate
    return list(dict.fromkeys(generated_sequences))


def run_frontier(
    args,
    protein: str,
    data_csv: str,
    value_col: str = "fitness",
    noise_std: float = 0.0,
    replicates: int = 1,
    initial_seed_size: int = 10,
):
    assert protein in VALID_PROTEINS, f"Unknown protein: {protein}. Must be one of {VALID_PROTEINS}"

    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    baseline_mode = getattr(args, 'baseline_mode', None)

    # Provide defaults to keep things contained and point to existing configs if missing
    if not getattr(args, 'model_config_location', None):
        args.model_config_location = "configs/model/PNPT_ESM2_650M_final.json"
    if not getattr(args, 'target_config_location', None):
        args.target_config_location = ensure_local_target_config(
            os.path.join("insilico_analysis", "dataset_oracle", protein), protein
        )
    if not getattr(args, 'model_name_suffix', None):
        args.model_name_suffix = f"{protein}_frontier_final"

    # If a zero-shot CSV is provided, use it as auxiliary label to match training expectations
    zs_path = getattr(args, 'zero_shot_fitness_predictions_location', None)
    if zs_path and os.path.exists(zs_path):
        args.augmentation = 'zero_shot_fitness_predictions_auxiliary_labels'
    else:
        # Disable zero-shot if path missing or not provided
        if zs_path and not os.path.exists(zs_path):
            print(f"Zero-shot file not found at {zs_path}; disabling zero-shot augmentation.")
        args.augmentation = None

    # Finalize config (handles zero-shot augmentation and directory setup)
    args = setup_config_and_paths(args)

    args.save_model_checkpoint = not args.do_not_save_model_checkpoint
    args.frozen_embedding_parameters = not args.fine_tune_model_embedding_parameters

    # Setup DB and table using finalized target_config
    connection = connect_db(args.tablespace_location, args.database_name)
    target_config = args.target_config
    target_names = [target_name for target_name in target_config]
    create_table(connection, args.table_name, target_names)

    # Determine main phenotype name
    phenotype_name = get_main_target_from_config(target_config)
    assert phenotype_name is not None, "Could not determine main target from target_config"
    agent_name = "agent_" + phenotype_name

    # Oracle and frontier
    wt = args.starting_sequence.replace("*", "")
    # Multi-task: collect any non-main, in-NPT-loss continuous targets to load as auxiliary oracle columns.
    # Their target_config key is the column name in the data file (e.g. "abundance", "binding").
    extra_value_cols = [
        t for t, cfg in target_config.items()
        if t != phenotype_name
        and t != 'zero_shot_fitness_predictions'
        and cfg.get('type') == 'continuous'
        and cfg.get('in_NPT_loss', False)
    ]
    oracle = Oracle(
        csv_path=data_csv,
        wt_sequence=wt,
        value_col=value_col,
        noise_std=noise_std,
        n_replicates=replicates,
        extra_value_cols=extra_value_cols,
    )
    # Inject WT into oracle so it is discoverable during search even if absent from dataset
    wt_fitness_val = getattr(args, 'wt_fitness', None)
    if wt_fitness_val is not None and wt not in oracle._map:
        oracle._map[wt] = float(wt_fitness_val)
    frontier = Frontier(dataset_csv=data_csv, seq_col="sequence", value_col=value_col, restrict_length=len(wt), wt_sequence=wt)

    # Build initial seed: sample initial_seed_size random sequences from the dataset
    seed_candidates = build_initial_seed(frontier, initial_seed_size)
    print(f"Seeding with {len(seed_candidates)} random dataset sequences")
    seed_df = pd.DataFrame({
        'sequence': seed_candidates,
        'segmented_sequence': [[s] for s in seed_candidates]
    })
    # Acquire seed measurements
    pheno = oracle.query(seed_candidates)
    init_sequences = list(pheno.keys())

    init_df = pd.DataFrame({'sequence': init_sequences})
    init_df['segmented_sequence'] = [[s] for s in init_df['sequence']]
    init_df['mutant'] = init_df['sequence'].apply(lambda x: generate_mutation_string(args.starting_sequence, x))
    # Determine zero-shot column name
    zs_col = getattr(args, 'zero_shot_fitness_predictions_var_name', None) or 'ESM2_650M'
    for target in target_names:
        if target == phenotype_name:
            init_df[target] = init_df['sequence'].map(pheno)
        elif target == 'zero_shot_fitness_predictions' and getattr(args, 'zero_shot_fitness_predictions_location', None) and os.path.exists(getattr(args, 'zero_shot_fitness_predictions_location')):
            zdict = get_zero_shot_fitness_predictions(
                [s for s in init_df['sequence']],
                args.zero_shot_fitness_predictions_location,
                zero_shot_prediction_name=zs_col
            )
            init_df[target] = init_df['sequence'].map(zdict)
        elif target in getattr(oracle, '_extra_maps', {}):
            # Multi-task auxiliary target: pull from oracle; NaN if unmeasured (NPT masks per-target).
            init_df[target] = init_df['sequence'].apply(lambda s: oracle.lookup_extra(s, target))
        else:
            init_df[target] = 0.0
    init_df['created_by'] = [agent_name] * len(init_df)
    add_sequence_data(connection, args.table_name, init_df)

    # Train initial model (skip for random and onehot_mlp baselines)
    surrogate = None
    if baseline_mode not in ('random', 'onehot_mlp', 'onehot_mlp_greedy'):
        model, target_processing = train_model(
            args=args,
            tablespace_location=args.tablespace_location,
            database_name=args.database_name,
            table_name=args.table_name,
            model_checkpoint_folder=args.model_checkpoint_folder,
            model_name='_'.join([args.model_name_suffix, 'round_0']),
            embeddings_location=getattr(args, 'embeddings_location', None),
            verbose=True
        )
    else:
        model = None
        target_processing = None
        if baseline_mode in ('onehot_mlp', 'onehot_mlp_greedy'):
            surrogate = OneHotMLPSurrogate(device='cuda' if torch.cuda.is_available() else 'cpu')
            train_seqs = [str(s).replace('*', '') for s in init_df['sequence']]
            train_fitness = [float(pheno.get(s, pheno.get(s.replace('*', '') + '*', 0.0))) for s in init_df['sequence']]
            surrogate.fit(train_seqs, train_fitness)

    # Active learning loop
    # WT is NOT pre-labeled: it is a normal acquirable sequence (present in the frontier graph
    # and the oracle), so it can be proposed, measured, trained on, and become a parent like any
    # other variant. It enters labeled_set only once actually acquired (see labeled_set.add below).
    labeled_set: Set[str] = set(s.replace("*", "") for s in init_df['sequence'])
    # Proposal mode: default to poisson_exact_n_enumerate (dataset-aware neighbor search)
    proposal_mode = getattr(args, 'proposal_mode', None) or 'poisson_exact_n_enumerate'

    round_log: List[dict] = []
    # Log seed as round 1
    for _, row in init_df.iterrows():
        round_log.append({'round': 1, 'sequence': row['sequence'], phenotype_name: row.get(phenotype_name, float('nan'))})

    num_samples = max(getattr(args, 'num_samples_per_iteration', 1000), args.acquisition_batch)

    for iteration_idx in range(1, args.num_acquisitions):
        print(f"STARTING ITER: {iteration_idx}")
        # Refresh training data
        train_data_df, target_processing = get_train_database(args, args.tablespace_location, args.database_name, args.table_name)

        # Propose candidates
        if proposal_mode == 'conditional_sampling':
            # Conditional sampling (label boosting + masked-AA sampling) to propose 1-step candidates
            candidate_sequences = conditional_sample_one_step_candidates(
                model=model,
                args=args,
                train_data_df=train_data_df,
                phenotype_name=phenotype_name,
                frontier=frontier,
                labeled_set=labeled_set,
                starting_sequence=args.starting_sequence,
                num_samples=max(getattr(args, 'num_samples_per_iteration', 0), args.acquisition_batch),
                target_processing=target_processing,
                max_positions_per_seq=None
            )

        elif proposal_mode == 'poisson_neighbors':
            # Enumerate neighbors per base sequence with a Poisson-sampled exact Hamming distance
            # Define base sequences: top quartile by phenotype, up to 30; else fall back to all training data
            q_threshold = train_data_df[phenotype_name].quantile(0.75)
            filtered = train_data_df[train_data_df[phenotype_name] >= q_threshold]
            if len(filtered) > 0:
                base_df = filtered.sample(min(30, len(filtered)), random_state=42)
            else:
                base_df = train_data_df.copy()
            base_sequences: List[str] = [str(s).replace('*', '') for s in base_df['sequence'].astype(str) if isinstance(s, str)]

            # Precompute Poisson pmf for lambda=1 and fold P(0) into P(1)
            lam = 1.0
            # Support k up to, say, 5 for practicality; neighbors beyond often sparse
            max_k = int(getattr(args, 'max_poisson_k', 5))
            k_values = np.arange(0, max_k + 1)
            pmf = np.exp(-lam) * np.power(lam, k_values) / np.array([math.factorial(int(k)) for k in k_values], dtype=float)
            if max_k >= 1:
                pmf[1] += pmf[0]
                pmf[0] = 0.0
            pmf = pmf / pmf.sum()

            candidate_set: Set[str] = set()
            for base_seq in base_sequences:
                # Sample an exact radius per base sequence (at least 1 due to folding)
                k = int(np.random.choice(k_values, p=pmf))
                if k <= 0:
                    k = 1
                # Enumerate exactly distance-k neighbors from this base, restrict to dataset via Frontier
                for neigh in frontier.neighbors_at_exact_distance(base_seq, k):
                    if neigh not in labeled_set:
                        candidate_set.add(neigh)
            candidate_sequences = [s + "*" for s in sorted(candidate_set)]
        elif proposal_mode == 'poisson_sampled_mutants':
            # For each base sequence in the top quartile (or up to 30), sample k ~ Poisson(1) with P(1)+=P(0),
            # then sample exactly ONE sequence at Hamming distance k from that base (if available), from dataset.
            q_threshold = train_data_df[phenotype_name].quantile(0.75)
            filtered = train_data_df[train_data_df[phenotype_name] >= q_threshold]
            if len(filtered) > 0:
                base_df = filtered.sample(min(30, len(filtered)), random_state=42)
            else:
                base_df = train_data_df.copy()
            base_sequences: List[str] = [str(s).replace('*', '') for s in base_df['sequence'].astype(str) if isinstance(s, str)]

            lam = 1.0
            max_k = int(getattr(args, 'max_poisson_k', 5))
            k_values = np.arange(0, max_k + 1)
            pmf = np.exp(-lam) * np.power(lam, k_values) / np.array([math.factorial(int(k)) for k in k_values], dtype=float)
            if max_k >= 1:
                pmf[1] += pmf[0]
                pmf[0] = 0.0
            pmf = pmf / pmf.sum()

            candidates: List[str] = []
            for base_seq in base_sequences:
                k = int(np.random.choice(k_values, p=pmf))
                if k <= 0:
                    k = 1
                neighbors_k = [n for n in frontier.neighbors_at_exact_distance(base_seq, k) if n not in labeled_set]
                if not neighbors_k:
                    continue
                # Sample one neighbor uniformly at this exact distance
                chosen = random.choice(neighbors_k)
                candidates.append(chosen + "*")
            candidate_sequences = sorted(list(dict.fromkeys(candidates)))
        elif proposal_mode == 'poisson_exact_n_enumerate':
            # Sample ONE global k ~ Poisson(1) with P(1)+=P(0), then enumerate ALL exact-k neighbors
            # for each base sequence in the top quartile (or up to 30). Deduplicate and exclude labeled.
            q_threshold = train_data_df[phenotype_name].quantile(0.75)
            filtered = train_data_df[train_data_df[phenotype_name] >= q_threshold]
            if len(filtered) > 0:
                base_df = filtered.sample(min(30, len(filtered)), random_state=42)
            else:
                base_df = train_data_df.copy()
            base_sequences: List[str] = [str(s).replace('*', '') for s in base_df['sequence'].astype(str) if isinstance(s, str)]

            lam = 1.0
            max_k = int(getattr(args, 'max_poisson_k', 5))
            k_values = np.arange(0, max_k + 1)
            pmf = np.exp(-lam) * np.power(lam, k_values) / np.array([math.factorial(int(k)) for k in k_values], dtype=float)
            if max_k >= 1:
                pmf[1] += pmf[0]
                pmf[0] = 0.0
            pmf = pmf / pmf.sum()

            k = int(np.random.choice(k_values, p=pmf))
            if k <= 0:
                k = 1

            candidate_set: Set[str] = set()
            for base_seq in base_sequences:
                for neigh in frontier.neighbors_at_exact_distance(base_seq, k):
                    if neigh not in labeled_set:
                        candidate_set.add(neigh)
            # Subsample to num_samples if candidate pool is too large (e.g. k=2 covers most of the dataset)
            candidate_list = sorted(candidate_set)
            if len(candidate_list) > num_samples:
                print(f"  Subsampling candidates: {len(candidate_list)} -> {num_samples}")
                candidate_list = random.sample(candidate_list, num_samples)
            candidate_sequences = [s + "*" for s in candidate_list]
        else:
            # Enumerate all neighbors within Hamming radius (default 1)
            hamming_radius = int(getattr(args, 'hamming_radius', 1))
            if hamming_radius <= 1:
                candidate_set = set(frontier.frontier(labeled_set))
            else:
                candidate_set = set(frontier.frontier_k(labeled_set, k=hamming_radius))
            candidate_sequences = [s + "*" for s in sorted(candidate_set)]

        # Score candidates
        if not candidate_sequences:
            print("Frontier empty; stopping early.")
            break

        if baseline_mode == 'random':
            # Random baseline: skip model scoring, randomly select
            n_select = min(args.acquisition_batch, len(candidate_sequences))
            sequences_to_acquire = random.sample(candidate_sequences, n_select)
        elif baseline_mode in ('onehot_mlp', 'onehot_mlp_greedy'):
            # One-hot MLP ensemble baseline: score with surrogate
            cand_seqs_clean = [s.replace('*', '') for s in candidate_sequences]
            mean_pred, std_pred = surrogate.predict(cand_seqs_clean)
            surrogate_df = pd.DataFrame({
                'sequence': candidate_sequences,
                'predictions_avg_' + phenotype_name: mean_pred,
                'uncertainty_' + phenotype_name: std_pred,
                'mutant': [generate_mutation_string(args.starting_sequence, s) for s in candidate_sequences],
            })
            alpha = 0.0 if baseline_mode == 'onehot_mlp_greedy' else 1.0
            sequences_to_acquire = select_mutants_to_acquire(
                mutated_sequences_pred=surrogate_df,
                phenotype_name=phenotype_name,
                num_sequences_to_acquire=args.acquisition_batch,
                acquisition_method='ucb',
                ucb_alpha=alpha,
                selection_method='top_n',
            )
            sequences_to_acquire = [seq + "*" if not seq.endswith("*") else seq for seq in sequences_to_acquire]
        else:
            # Full pipeline or greedy: score with ProteinNPT
            test_data_df = pd.DataFrame({
                'sequence': candidate_sequences,
                'segmented_sequence': [[s] for s in candidate_sequences]
            })
            test_data_df['mutant'] = test_data_df['sequence'].apply(lambda x: generate_mutation_string(args.starting_sequence, x))
            train_data_df = data_cleanup_for_model_input(train_data_df)
            test_data_df = data_cleanup_for_model_input(test_data_df)
            add_back_embedding(model)
            sampled_sequences_pred = score_mutated_sequences(
                model=model,
                test_data_df=test_data_df,
                training_data_df=train_data_df,
                target_processing=target_processing,
                uncertainty_method=args.uncertainty_method,
                num_MC_dropout_samples=args.num_MC_dropout_samples,
                return_uncertainty=True
            )

            # Greedy: ucb_alpha=0 (no exploration); full pipeline: ucb_alpha=1
            alpha = 0.0 if baseline_mode == 'greedy' else 1.0
            sequences_to_acquire = select_mutants_to_acquire(
                mutated_sequences_pred=sampled_sequences_pred,
                phenotype_name=phenotype_name,
                num_sequences_to_acquire=args.acquisition_batch,
                acquisition_method=args.acquisition_method,
                ucb_alpha=alpha,
                selection_method=args.selection_method,
                ESM_location=args.ESM_location
            )
            sequences_to_acquire = [seq + "*" if not seq.endswith("*") else seq for seq in sequences_to_acquire]

        # Acquire from oracle
        acquired = oracle.query(sequences_to_acquire)
        if not acquired:
            print("No valid sequences acquired from oracle in this round.")
            continue

        new_df = pd.DataFrame({'sequence': list(acquired.keys())})
        new_df['segmented_sequence'] = [[s] for s in new_df['sequence']]
        new_df['mutant'] = new_df['sequence'].apply(lambda x: generate_mutation_string(args.starting_sequence, x))
        for target in target_names:
            if target == phenotype_name:
                new_df[target] = new_df['sequence'].map(acquired)
            elif target == 'zero_shot_fitness_predictions' and getattr(args, 'zero_shot_fitness_predictions_location', None) and os.path.exists(getattr(args, 'zero_shot_fitness_predictions_location')):
                zdict = get_zero_shot_fitness_predictions(
                    [s for s in new_df['sequence']],
                    args.zero_shot_fitness_predictions_location,
                    zero_shot_prediction_name=zs_col
                )
                new_df[target] = new_df['sequence'].map(zdict)
            elif target in getattr(oracle, '_extra_maps', {}):
                new_df[target] = new_df['sequence'].apply(lambda s: oracle.lookup_extra(s, target))
            else:
                new_df[target] = 0.0
        new_df['created_by'] = [agent_name] * len(new_df)
        add_sequence_data(connection, args.table_name, new_df)

        # Update labeled set and round log
        for seq in new_df['sequence']:
            labeled_set.add(seq.replace("*", ""))
        for _, row in new_df.iterrows():
            round_log.append({'round': iteration_idx + 1, 'sequence': row['sequence'], phenotype_name: row.get(phenotype_name, float('nan'))})

        # Retrain model (except maybe last iteration, but keeping consistent)
        if iteration_idx < args.num_acquisitions - 1:
            if baseline_mode not in ('random', 'onehot_mlp', 'onehot_mlp_greedy'):
                model, target_processing = train_model(
                    args=args,
                    tablespace_location=args.tablespace_location,
                    database_name=args.database_name,
                    table_name=args.table_name,
                    model_checkpoint_folder=args.model_checkpoint_folder,
                    model_name='_'.join([args.model_name_suffix, 'round', str(iteration_idx)]),
                    embeddings_location=getattr(args, 'embeddings_location', None),
                    verbose=True
                )
            elif baseline_mode in ('onehot_mlp', 'onehot_mlp_greedy'):
                # Retrain MLP ensemble on all labeled data
                all_train_df, _ = get_train_database(args, args.tablespace_location, args.database_name, args.table_name)
                train_seqs = [str(s).replace('*', '') for s in all_train_df['sequence']]
                train_fitness = all_train_df[phenotype_name].astype(float).tolist()
                surrogate.fit(train_seqs, train_fitness)
        print(f"Completed round {iteration_idx + 1}/{args.num_acquisitions}")

    # Final results sorted by phenotype
    final_df, _ = get_train_database(args, args.tablespace_location, args.database_name, args.table_name)
    final_df = final_df.sort_values(by=phenotype_name, ascending=False).reset_index(drop=True)

    # Save round log
    round_log_df = pd.DataFrame(round_log)
    return final_df, round_log_df


if __name__ == "__main__":
    # Minimal CLI that relies on the same arguments as self_driving.py provided by a JSON args file
    parser = argparse.ArgumentParser(description='Run dataset oracle frontier simulation (unified across proteins)')
    parser.add_argument('--protein', type=str, required=True, choices=list(VALID_PROTEINS), help='Protein to simulate')
    parser.add_argument('--args_json', type=str, required=True, help='Path to a JSON file with fields matching self_driving.parse_arguments output')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to dataset CSV/PKL with columns sequence, fitness')
    parser.add_argument('--value_col', type=str, default='functional_score')
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--initial_seed_size', type=int, default=10)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--wt_fitness', type=float, default=None)
    parser.add_argument('--proposal_mode', type=str, default=None, help="'frontier_neighbors' or 'conditional_sampling'")
    parser.add_argument('--hamming_radius', type=int, default=1, help='Frontier neighbor radius (1 = singles, 2 = include doubles, etc.)')
    parser.add_argument('--max_poisson_k', type=int, default=5, help='Maximum k to consider when sampling Poisson radii')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--baseline_mode', type=str, default=None, choices=['random', 'greedy', 'onehot_mlp', 'onehot_mlp_greedy'], help='Baseline mode (None = full pipeline)')
    args_cli = parser.parse_args()

    # Load args namespace from JSON
    with open(args_cli.args_json, 'r') as f:
        args_dict = json.load(f)
    args_ns = argparse.Namespace(**args_dict)
    # Attach CLI-provided overrides
    args_ns.wt_fitness = args_cli.wt_fitness
    args_ns.seed = args_cli.seed
    args_ns.baseline_mode = args_cli.baseline_mode

    # Attach CLI-provided overrides
    if args_cli.proposal_mode is not None:
        args_ns.proposal_mode = args_cli.proposal_mode
    if args_cli.hamming_radius is not None:
        args_ns.hamming_radius = args_cli.hamming_radius
    if args_cli.max_poisson_k is not None:
        args_ns.max_poisson_k = args_cli.max_poisson_k

    results, round_log_df = run_frontier(
        args=args_ns,
        protein=args_cli.protein,
        data_csv=args_cli.data_csv,
        value_col=args_cli.value_col,
        noise_std=args_cli.noise_std,
        replicates=args_cli.replicates,
        initial_seed_size=args_cli.initial_seed_size,
    )

    if args_cli.output_csv:
        os.makedirs(os.path.dirname(args_cli.output_csv), exist_ok=True)
        results.to_csv(args_cli.output_csv, index=False)
        # Save round log alongside results
        round_log_path = args_cli.output_csv.replace('.csv', '_round_log.csv')
        round_log_df.to_csv(round_log_path, index=False)
    else:
        print(results.head())
