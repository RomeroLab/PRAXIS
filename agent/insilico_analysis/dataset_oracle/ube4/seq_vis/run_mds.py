#!/usr/bin/env python3
"""Compute a 2D MDS embedding for the UBE4 sequence/function dataset.

This script performs classical multidimensional scaling (MDS) on the
`ube4_SeqFxnDataset.pkl` file by leveraging the equivalence between
Euclidean-distance MDS and principal component analysis. Sequences are
expanded into a one-hot representation, an IncrementalPCA fit produces the
low-dimensional coordinates, and the resulting embedding along with each
sequence and functional score are written to CSV.
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDS embedding on UBE4 sequences.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("insilico_analysis/dataset_oracle/ube4/ube4_SeqFxnDataset.pkl"),
        help="Path to the SeqFxnDataset pickle file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("insilico_analysis/dataset_oracle/ube4/seq_vis/ube4_mds_embedding.csv"),
        help="Path to the CSV file that will store the embedding.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of embedding dimensions to compute (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of sequences processed per batch during IncrementalPCA.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of sequences to embed (for testing).",
    )
    parser.add_argument(
        "--wt-sequence",
        type=str,
        default=None,
        help="Optional wild-type sequence to inject if missing from the dataset.",
    )
    parser.add_argument(
        "--wt-functional-score",
        type=float,
        default=None,
        help="Functional score to assign to the wild-type sequence (default: NaN).",
    )
    return parser.parse_args()


def batched(data: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    """Yield consecutive slices of *data* of length at most *batch_size*."""

    n = len(data)
    for start in range(0, n, batch_size):
        yield data[start : start + batch_size]


def sequences_to_index_matrix(
    sequences: Sequence[str],
    letter_to_index: dict[str, int],
    seq_length: int,
) -> np.ndarray:
    """Convert a batch of sequences to an integer matrix of shape (batch, length)."""

    batch_size = len(sequences)
    indices = np.empty((batch_size, seq_length), dtype=np.int16)
    for row, seq in zip(indices, sequences):
        if len(seq) != seq_length:
            raise ValueError(f"Sequence length mismatch: expected {seq_length}, got {len(seq)}")
        try:
            row[:] = [letter_to_index[ch] for ch in seq]
        except KeyError as err:
            raise KeyError(f"Unexpected residue {err.args[0]!r} encountered in sequence.") from err
    return indices


def encode_batch(
    sequences: Sequence[str],
    letter_to_index: dict[str, int],
    seq_length: int,
    eye: np.ndarray,
) -> np.ndarray:
    """Return a dense one-hot encoding for a batch of sequences."""

    if not sequences:
        return np.empty((0, seq_length * eye.shape[0]), dtype=np.float32)
    index_matrix = sequences_to_index_matrix(sequences, letter_to_index, seq_length)
    return eye[index_matrix].reshape(len(sequences), seq_length * eye.shape[0])


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(input_path)
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    if "sequence" not in df.columns or "functional_score" not in df.columns:
        raise KeyError("Input dataframe must contain 'sequence' and 'functional_score' columns.")

    wt_sequence: str | None = None
    if args.wt_sequence is not None:
        wt_sequence = args.wt_sequence.strip()
        if not wt_sequence:
            wt_sequence = None

    if wt_sequence is not None and wt_sequence not in set(df["sequence"].astype(str)):
        wt_row = {col: np.nan for col in df.columns}
        wt_row["sequence"] = wt_sequence
        if "functional_score" in wt_row and args.wt_functional_score is not None:
            wt_row["functional_score"] = args.wt_functional_score
        df = pd.concat([df, pd.DataFrame([wt_row])], ignore_index=True)
        print("Inserted provided wild-type sequence into dataset for embedding.")

    sequences = df["sequence"].astype(str).tolist()
    if not sequences:
        raise ValueError("No sequences found in the input dataset.")

    seq_lengths = df["sequence"].str.len().unique()
    if len(seq_lengths) != 1:
        raise ValueError(f"Sequences have varying lengths: {sorted(seq_lengths)}")
    seq_length = int(seq_lengths[0])

    if wt_sequence is not None and wt_sequence not in sequences:
        raise ValueError("Wild-type sequence injection failed.")

    alphabet = sorted({ch for seq in sequences for ch in seq})
    if not alphabet:
        raise ValueError("Alphabet could not be determined from sequences.")

    letter_to_index = {letter: idx for idx, letter in enumerate(alphabet)}
    identity = np.eye(len(alphabet), dtype=np.float32)

    n_components = args.n_components
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")
    if n_components > len(alphabet) * seq_length:
        raise ValueError("n_components exceeds the flattened feature dimension.")

    batch_size = max(1, args.batch_size)

    print(
        f"Fitting IncrementalPCA (classical MDS) on {len(sequences)} sequences "
        f"(length {seq_length}, alphabet size {len(alphabet)})."
    )

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # First pass: fit the IncrementalPCA model.
    num_batches = ceil(len(sequences) / batch_size)

    for batch in tqdm(
        batched(sequences, batch_size),
        total=num_batches,
        desc="Fitting IPCA",
    ):
        encoded = encode_batch(batch, letter_to_index, seq_length, identity)
        ipca.partial_fit(encoded)

    # Second pass: transform to obtain low-dimensional coordinates.
    embedding = np.zeros((len(sequences), n_components), dtype=np.float32)
    offset = 0
    for batch in tqdm(
        batched(sequences, batch_size),
        total=num_batches,
        desc="Transforming",
    ):
        encoded = encode_batch(batch, letter_to_index, seq_length, identity)
        transformed = ipca.transform(encoded)
        end = offset + len(transformed)
        embedding[offset:end] = transformed
        offset = end

    result = pd.DataFrame(
        {
            "sequence": sequences,
            **{f"mds_dim{i+1}": embedding[:, i] for i in range(n_components)},
            "functional_score": df["functional_score"].values,
        }
    )

    result.to_csv(output_path, index=False)
    print(f"Wrote embedding for {len(result)} sequences to {output_path}.")


if __name__ == "__main__":
    main()

