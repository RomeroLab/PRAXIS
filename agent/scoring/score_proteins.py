#!/usr/bin/env python
"""
ProteinNPT scorer (PRAXIS-integrated).

Scores a CSV of protein amino-acid sequences with a trained ProteinNPT
checkpoint (a ``checkpoint.t7`` file) and writes out predicted a1 / a2 / a3
values. It reproduces the scoring path used inside the self_driving / PRAXIS
pipeline (``score_mutated_sequences`` in ``agent/utils.py``).

This lives inside the PRAXIS repo and reuses the pipeline's own code rather than
vendoring copies: the ConvBert compat shim, ``data_cleanup_for_model_input``,
``add_back_embedding`` and -- crucially -- ``get_train_database`` all come from
``agent/utils.py``.

Labelled context (required)
---------------------------
ProteinNPT is a *retrieval / in-context* model: to predict the labels of a query
sequence it is fed a set of already-labelled "training" sequences in the same
forward pass (their real labels are visible, the query's labels are masked and
predicted). The checkpoint stores only model weights, so the labelled context
must be supplied separately.

By default the context comes from the assayed-sequence **SQLite database** in
``agent/data`` (the same DB the pipeline reads via ``get_train_database``): if a
single user table is present it is auto-detected, otherwise pass ``--table``.
Pass ``--train_csv`` to override the DB with an ad-hoc labelled CSV instead.

Either way the continuous targets are z-scored by their own mean/std before being
fed as context (matching training). By default predictions are un-normalized back
to raw label units; ``--normalized_output`` keeps them in z-scored space.

Example
-------
    python scoring/score_proteins.py \
        --input_csv scoring/_test_input.csv \
        --checkpoint /path/to/<name>/final/checkpoint.t7 \
        --output_csv scored.csv
    # context: agent/data/Bgl_spec_2026_final_database_final.db (auto-detected table)
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
import torch
import tqdm
from datasets import Dataset

# Put agent/ (the parent dir) on sys.path so the bare pipeline imports resolve,
# mirroring how insilico_analysis/ reaches its siblings. Must precede the
# proteinnpt / utils imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Re-export ConvBertLayer onto transformers so proteinnpt imports resolve; must
# run before importing anything from proteinnpt (utils does this too, but importing
# it explicitly keeps this script correct regardless of import ordering).
import _compat_proteinnpt  # noqa: F401,E402

from proteinnpt.proteinnpt.model import ProteinNPTModel  # noqa: E402
from proteinnpt.proteinnpt.data_processing import process_batch  # noqa: E402
from proteinnpt.utils.data_utils import (  # noqa: E402
    collate_fn_protein_npt,
    preprocess_training_targets,
)
from proteinnpt.utils.esm.data import Alphabet  # noqa: E402

# Reuse the pipeline's own helpers instead of vendoring copies.
from utils import (  # noqa: E402
    data_cleanup_for_model_input,
    add_back_embedding,
    get_train_database,
    fetch_data_for_training,
    connect_db,
)

_AGENT_DIR = Path(__file__).resolve().parents[1]

# ESM2-650M weights (esm2_t33_650M_UR50D.pt, with its *-contact-regression.pt
# sibling alongside). Defaults to the copy shipped under agent/ESM; override with
# the ESM2_650M_PATH env var or --esm_location. The path baked into some
# checkpoints is stale, which is why load_model overwrites it.
DEFAULT_ESM_LOCATION = os.environ.get(
    "ESM2_650M_PATH",
    str(_AGENT_DIR / "ESM" / "ESM2" / "esm2_t33_650M_UR50D.pt"),
)

# Default labelled-context DB: the assayed-sequence SQLite database in agent/data.
DEFAULT_DATABASE = str(_AGENT_DIR / "data" / "Bgl_spec_2026_final_database_final.db")


# --------------------------------------------------------------------------- #
# Scoring (trimmed port of utils.py::score_mutated_sequences for the ESM /
# non-MSA / non-covariate case used by the self_driving models, made
# device-agnostic -- the model is assumed already on ``model.device``).
# --------------------------------------------------------------------------- #
def score_mutated_sequences(model, test_data_df, training_data_df, target_processing,
                            return_uncertainty=False, num_MC_dropout_samples=10,
                            selected_indices_seed=0, verbose=True):
    """
    Returns a DataFrame with columns ``predictions_avg_<target>`` (and
    ``uncertainty_<target>`` when requested) plus ``mutant`` and ``sequence``.
    Row order matches ``test_data_df`` (the eval loader does not shuffle).
    """
    model.eval()
    # Ensure training targets are numeric floats to avoid numpy.object_ arrays downstream.
    for target_name in getattr(model, 'target_names', []) or []:
        if target_name in training_data_df.columns:
            training_data_df[target_name] = pd.to_numeric(
                training_data_df[target_name], errors='coerce').astype(float)

    test_dataset = Dataset.from_pandas(test_data_df)
    train_dataset = Dataset.from_pandas(training_data_df)
    model_output = defaultdict(list)
    targets_to_predict = model.target_names
    with torch.no_grad():
        eval_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=model.args.eval_num_sequences_to_score_per_batch_per_gpu,
            shuffle=False,
            num_workers=model.args.num_data_loaders_workers,
            pin_memory=True,
            collate_fn=collate_fn_protein_npt,
        )
        iterable = tqdm.tqdm(eval_loader, desc="Scoring sequences") if verbose else eval_loader
        for batch in iterable:
            processed_batch = process_batch(
                batch=batch,
                model=model,
                alphabet=model.alphabet,
                args=model.args,
                MSA_sequences=None,
                MSA_weights=None,
                MSA_start_position=None,
                MSA_end_position=None,
                target_processing=target_processing,
                training_sequences=train_dataset,
                proba_target_mask=1.0,
                proba_aa_mask=0.0,
                eval_mode=True,
                device=model.device,
                selected_indices_seed=selected_indices_seed,
                indel_mode=model.args.indel_mode,
            )
            # self_driving models use the 'auxiliary_labels' augmentation, so
            # zero-shot predictions are handled as an ordinary (masked) target and
            # are not passed to forward as a covariate.
            zero_shot_fitness_predictions = None
            if model.args.augmentation == "zero_shot_fitness_predictions_covariate":
                zero_shot_fitness_predictions = processed_batch['target_labels'][
                    'zero_shot_fitness_predictions'].view(-1, 1)
                del processed_batch['target_labels']['zero_shot_fitness_predictions']

            num_of_mutated_seqs_to_score = processed_batch['num_of_mutated_seqs_to_score']

            if not return_uncertainty:
                output = model.forward(
                    tokens=processed_batch['masked_tokens'],
                    targets=processed_batch['masked_targets'],
                    zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                    sequence_embeddings=processed_batch['sequence_embeddings'],
                )
                for target_name in targets_to_predict:
                    model_output["predictions_avg_" + target_name] += list(
                        output["target_predictions"][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
            else:
                output = model.forward_with_uncertainty(
                    tokens=processed_batch['masked_tokens'],
                    targets=processed_batch['masked_targets'],
                    zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                    sequence_embeddings=processed_batch['sequence_embeddings'],
                    num_MC_dropout_samples=num_MC_dropout_samples,
                )
                for target_name in targets_to_predict:
                    model_output["predictions_avg_" + target_name] += list(
                        output[target_name]["predictions_avg"][:num_of_mutated_seqs_to_score].cpu().numpy())
                    model_output["uncertainty_" + target_name] += list(
                        output[target_name]["uncertainty"][:num_of_mutated_seqs_to_score].cpu().numpy())
            model_output["mutant_mutated_seq_pairs"] += \
                processed_batch["mutant_mutated_seq_pairs"][:num_of_mutated_seqs_to_score]

    model_output["mutant"], model_output["sequence"] = zip(*model_output["mutant_mutated_seq_pairs"])
    del model_output["mutant_mutated_seq_pairs"]
    return pd.DataFrame(model_output)


# --------------------------------------------------------------------------- #
# Context loading + plumbing
# --------------------------------------------------------------------------- #
def continuous_targets(target_config):
    """Target names treated as continuous (standard-scaled) by proteinnpt.

    Matches preprocess_training_targets: any target of type 'continuous', plus
    'zero_shot_fitness_predictions' which is always scaled regardless of config.
    """
    names = [n for n, cfg in target_config.items() if cfg.get("type") == "continuous"]
    if "zero_shot_fitness_predictions" in target_config and \
            "zero_shot_fitness_predictions" not in names:
        names.append("zero_shot_fitness_predictions")
    return names


def detect_table(db_path):
    """Return the single user table in a SQLite DB, or exit asking for --table."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        connection.close()
    names = [r[0] for r in rows]
    if len(names) == 1:
        return names[0]
    if not names:
        sys.exit(f"No user tables found in {db_path}.")
    sys.exit(f"Multiple tables in {db_path}: {names}. Disambiguate with --table.")


def _canonical_categories(canonical_db, table_name, target_config):
    """Full set of values each categorical target takes in the canonical (full) DB.

    proteinnpt's preprocess_training_targets derives a categorical target's
    index mapping from ``set(values)`` in whatever data it is handed, and asserts
    the count equals the configured ``dim``. A truncated early-round context can
    be single-valued (e.g. every seed sequence has f=1.0), which both trips that
    assertion and -- if bypassed naively -- would map that lone value to index 0
    instead of the index the full-data (training) mapping gives it. Sourcing the
    category set from the full DB reproduces the exact same ``set`` the non-
    truncated rounds (and training) use, so every round encodes consistently.
    """
    cats = {}
    con = sqlite3.connect(canonical_db)
    try:
        for name, cfg in target_config.items():
            if cfg.get("type") == "categorical":
                rows = con.execute(
                    f"SELECT DISTINCT {name} FROM {table_name} "
                    f"WHERE {name} IS NOT NULL").fetchall()
                cats[name] = set(r[0] for r in rows)
    finally:
        con.close()
    return cats


def _preprocess_targets(raw_targets, target_config, categorical_categories):
    """preprocess_training_targets, but categorical mappings use an externally
    supplied (full-DB) category set instead of this slice's values -- so a
    single-valued early-round categorical still encodes consistently and doesn't
    trip the dim assertion. Continuous/zero-shot targets are z-scored by their own
    context-local mean/std with the same zero-variance epsilon guard as
    get_train_database. Numerically identical to the stock path whenever the slice
    already contains every category.
    """
    target_processing = {}
    for name in list(raw_targets.keys()):
        cfg = target_config.get(name, {})
        is_continuous = (cfg.get("type") == "continuous") or (name == "zero_shot_fitness_predictions")
        if is_continuous:
            arr = np.asarray(raw_targets[name], dtype=float)
            if np.nanstd(arr) == 0:  # epsilon guard (matches get_train_database)
                print(f"Warning: target '{name}' has zero variance. "
                      f"Applying epsilon guard for normalization.")
                arr = arr + np.linspace(0, 1e-8, len(arr))
            mean, std = np.nanmean(arr), np.nanstd(arr)
            target_processing[name] = {
                "mean": mean, "std": std,
                "P95": np.nanquantile(arr, q=0.95),
            }
            raw_targets[name] = pd.Series((arr - mean) / std,
                                          index=raw_targets[name].index)
        else:
            cats = categorical_categories.get(name)
            if not cats:  # fall back to this slice's own values
                cats = set(pd.Series(raw_targets[name]).dropna().unique())
            category_to_index = {c: i for i, c in enumerate(cats)}
            index_to_category = {i: c for i, c in enumerate(cats)}
            category_to_index['<mask>'] = len(cats)
            index_to_category[len(cats)] = '<mask>'
            raw_targets[name] = torch.tensor(
                [category_to_index[v] for v in raw_targets[name]])
            target_processing[name] = {
                "category_to_index": category_to_index,
                "index_to_category": index_to_category,
            }
    return raw_targets, target_processing


def load_context_from_db(model_args, db_path, table_name, canonical_db=None):
    """
    Load the labelled context from the assayed-sequence SQLite DB. Mirrors the
    pipeline's get_train_database (same fetch + continuous z-scoring), but derives
    categorical target mappings from the full/canonical DB so truncated early-round
    slices don't break on single-valued categoricals (see _canonical_categories).
    Returns (train_df, target_processing). ``canonical_db`` defaults to the full
    assayed-sequence DB shipped in agent/data.
    """
    if not os.path.exists(db_path):
        sys.exit(f"Context database not found: {db_path}\n"
                 f"Point at it with --database, or supply a CSV with --train_csv.")
    table_name = table_name or detect_table(db_path)
    canonical_db = canonical_db or DEFAULT_DATABASE
    print(f"Context DB: {os.path.basename(db_path)} (table '{table_name}')")
    target_names = list(model_args.target_config.keys())
    connection = connect_db(
        tablespace_location=os.path.dirname(os.path.abspath(db_path)),
        database_name=os.path.basename(db_path),
    )
    df = fetch_data_for_training(connection, table_name, target_names)
    categorical_categories = _canonical_categories(
        canonical_db, table_name, model_args.target_config)
    raw_targets = {name: df[name] for name in target_names}
    preprocessed, target_processing = _preprocess_targets(
        raw_targets, model_args.target_config, categorical_categories)
    for name in target_names:
        df[name] = preprocessed[name]
    return df, target_processing


def load_context_from_csv(train_csv, target_config):
    """
    Override path: load labelled context from an ad-hoc CSV and normalize it the
    same way get_train_database does (epsilon guard on zero-variance continuous
    targets + preprocess_training_targets). Returns (train_df, target_processing).
    """
    df = pd.read_csv(train_csv)
    target_names = list(target_config.keys())
    missing = [t for t in target_names if t not in df.columns]
    if missing:
        sys.exit(f"--train_csv is missing target columns required by this "
                 f"checkpoint: {missing}")
    raw_targets = {name: df[name] for name in target_names}
    # Epsilon guard matching get_train_database: a zero-variance continuous target
    # would divide by std=0 in preprocess_training_targets and produce NaN.
    for name in target_names:
        if target_config.get(name, {}).get("type") == "continuous":
            arr = raw_targets[name].to_numpy(dtype=float)
            if np.nanstd(arr) == 0:
                print(f"Warning: target '{name}' has zero variance. "
                      f"Applying epsilon guard for normalization.")
                jitter = np.linspace(0, 1e-8, len(arr))
                raw_targets[name] = pd.Series(arr + jitter, index=raw_targets[name].index)
    preprocessed, target_processing = preprocess_training_targets(raw_targets, target_config)
    for name in target_names:
        df[name] = preprocessed[name]
    return df, target_processing


def load_model(checkpoint_path, esm_location, device):
    """Rebuild a ProteinNPTModel from a checkpoint.t7 and put it on ``device``."""
    if not os.path.exists(esm_location):
        raise FileNotFoundError(
            f"ESM embedding weights not found at: {esm_location}\n"
            f"Pass --esm_location pointing to esm2_t33_650M_UR50D.pt "
            f"(the *-contact-regression.pt sibling must be alongside it)."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint['args']

    # The path baked into the checkpoint may be stale; ProteinNPTModel.__init__ and
    # add_back_embedding both load ESM from args.embedding_model_location, so fix it
    # here before either runs.
    args.embedding_model_location = esm_location
    args.sequence_embeddings_location = None  # compute embeddings on the fly
    args.num_data_loaders_workers = 0

    if args.aa_embeddings == "MSA_Transformer":
        alphabet = Alphabet.from_architecture("msa_transformer")
    else:  # ESM2 / ESM-1b share the ESM-1b alphabet
        alphabet = Alphabet.from_architecture("ESM-1b")

    if args.model_type != "ProteinNPT":
        raise NotImplementedError(f"Unsupported model_type: {args.model_type}")

    model = ProteinNPTModel(args, alphabet)
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Frozen-embedding checkpoints legitimately omit aa_embedding.* (reloaded from
    # ESM below). Anything else missing/unexpected is a real problem.
    real_missing = [k for k in missing if not k.startswith("aa_embedding")]
    if real_missing:
        raise RuntimeError(f"Missing non-embedding weights in checkpoint: {real_missing[:10]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected weights in checkpoint: {unexpected[:10]} ...")

    add_back_embedding(model)  # reloads from args.embedding_model_location set above
    model.to(device)
    model.set_device()
    model.eval()
    return model, args


def main():
    parser = argparse.ArgumentParser(
        description="Score protein sequences with a ProteinNPT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv", required=True,
                        help="CSV of sequences to score.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the ProteinNPT checkpoint.t7 file.")
    parser.add_argument("--output_csv", required=True,
                        help="Where to write sequences + predicted a1/a2/a3.")
    parser.add_argument("--database", default=DEFAULT_DATABASE,
                        help="SQLite context DB of assayed sequences.")
    parser.add_argument("--table", default=None,
                        help="Table in --database. Auto-detected if a single "
                             "user table is present.")
    parser.add_argument("--canonical_db", default=None,
                        help="Full DB used only to fix categorical target category "
                             "sets when --database is a truncated slice (defaults "
                             "to the shipped full assayed-sequence DB).")
    parser.add_argument("--train_csv", default=None,
                        help="Override the DB with an ad-hoc labelled context CSV "
                             "(must contain this checkpoint's target columns).")
    parser.add_argument("--esm_location", default=DEFAULT_ESM_LOCATION,
                        help="Path to esm2_t33_650M_UR50D.pt.")
    parser.add_argument("--seq_column", default="sequence",
                        help="Name of the amino-acid column in --input_csv.")
    parser.add_argument("--targets", default="a1,a2,a3",
                        help="Comma-separated targets to write out.")
    parser.add_argument("--normalized_output", action="store_true",
                        help="Leave predictions in z-scored units. By default the "
                             "script un-normalizes them back to the context's raw "
                             "label units (using its own per-target mean/std).")
    parser.add_argument("--return_uncertainty", action="store_true",
                        help="Also emit MC-dropout uncertainty per target "
                             "(note: these models train with dropout=0).")
    parser.add_argument("--num_MC_dropout_samples", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Override the per-batch chimera count "
                             "(eval_num_sequences_to_score_per_batch_per_gpu). Lower "
                             "it to fit large-context / small-GPU runs in memory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # ---- Load model ----
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, args.esm_location, device)
    if args.eval_batch_size is not None:
        model.args.eval_num_sequences_to_score_per_batch_per_gpu = args.eval_batch_size
        print(f"Eval batch size overridden -> {args.eval_batch_size}")
    print(f"Model targets: {model.target_names} | indel_mode={model_args.indel_mode}")

    # ---- Build labelled context (already-normalized) + target_processing ----
    # target_config comes from the checkpoint: the model was trained against it.
    if args.train_csv:
        print(f"Context: {args.train_csv} (--train_csv override)")
        train_df, target_processing = load_context_from_csv(
            args.train_csv, model_args.target_config)
    else:
        train_df, target_processing = load_context_from_db(
            model_args, args.database, args.table, args.canonical_db)
    print(f"Context set: {len(train_df)} labelled sequences")

    # get_train_database / the CSV loader both return mutant + segmented_sequence,
    # but guard anyway so an ad-hoc CSV without them still works.
    if 'mutant' not in train_df.columns:
        train_df['mutant'] = ""
    if 'segmented_sequence' not in train_df.columns:
        train_df['segmented_sequence'] = train_df['sequence']
    train_df_cleaned = data_cleanup_for_model_input(train_df, del_vars=True)

    # ---- Prepare query sequences ----
    input_df = pd.read_csv(args.input_csv)
    if args.seq_column not in input_df.columns:
        sys.exit(f"Column '{args.seq_column}' not in {args.input_csv}. "
                 f"Found: {list(input_df.columns)}")
    input_df[args.seq_column] = input_df[args.seq_column].astype(str).str.strip()

    test_df = pd.DataFrame({
        'sequence': input_df[args.seq_column].values,
        'mutant': "",                                   # identifier only (indel_mode)
        'segmented_sequence': input_df[args.seq_column].values,  # unused; dropped
    })
    test_df_cleaned = data_cleanup_for_model_input(test_df, del_vars=True)

    # ---- Score ----
    print(f"Scoring {len(input_df)} sequences on {device} ...")
    predictions = score_mutated_sequences(
        model=model,
        test_data_df=test_df_cleaned,
        training_data_df=train_df_cleaned,
        target_processing=target_processing,
        return_uncertainty=args.return_uncertainty,
        num_MC_dropout_samples=args.num_MC_dropout_samples,
        verbose=True,
    )

    assert len(predictions) == len(input_df), \
        f"Prediction/input row mismatch: {len(predictions)} vs {len(input_df)}"

    # ---- Assemble output ----
    # The model predicts in the z-scored space defined by the context's mean/std.
    # By default we un-normalize back to raw label units (pred_raw = pred_z * std +
    # mean); --normalized_output keeps the z-scores. Un-normalizing is
    # self-consistent: the same per-target mean/std that scaled the context is
    # applied inversely to its predictions.
    continuous = set(continuous_targets(model_args.target_config))
    out = input_df.copy()
    out["sequence"] = input_df[args.seq_column].values
    for target_name in output_targets:
        col = "predictions_avg_" + target_name
        if col not in predictions.columns:
            sys.exit(f"Target '{target_name}' not produced by model "
                     f"(available: {list(model.target_names)}).")
        pred = np.asarray(predictions[col].values, dtype=float)
        unc = None
        if args.return_uncertainty and ("uncertainty_" + target_name) in predictions.columns:
            unc = np.asarray(predictions["uncertainty_" + target_name].values, dtype=float)
        if not args.normalized_output and target_name in continuous:
            std = target_processing[target_name]['std']
            mean = target_processing[target_name]['mean']
            pred = pred * std + mean
            if unc is not None:
                unc = unc * std  # scale-only for a std-dev-like uncertainty
        out[target_name] = pred
        if unc is not None:
            out[target_name + "_uncertainty"] = unc

    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} rows -> {args.output_csv}")
    units = ("normalized (z-scored)" if args.normalized_output
             else "the context's raw label units (un-normalized)")
    print(f"Note: predicted {','.join(output_targets)} are in {units}.")


if __name__ == "__main__":
    main()
