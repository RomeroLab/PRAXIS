#!/usr/bin/env python3
"""Plot the UBE4 MDS embedding as a scatter plot colored by functional score."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MDS scatter plot for UBE4.")
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(
            "insilico_analysis/dataset_oracle/ube4/seq_vis/ube4_mds_embedding.csv"
        ),
        help="CSV produced by run_mds.py containing sequence coordinates and scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "insilico_analysis/dataset_oracle/ube4/seq_vis/ube4_mds_scatter.png"
        ),
        help="Destination path for the scatter plot image (PNG).",
    )
    parser.add_argument(
        "--x-col",
        default="mds_dim1",
        help="Column name for the x-axis coordinate.",
    )
    parser.add_argument(
        "--y-col",
        default="mds_dim2",
        help="Column name for the y-axis coordinate.",
    )
    parser.add_argument(
        "--color-col",
        default="functional_score",
        help="Column used to color points.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows to randomly sample before plotting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots-per-inch for the saved figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_path = args.embedding.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(embedding_path)

    required_cols = {args.x_col, args.y_col, args.color_col}
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Embedding file missing required columns: {missing_str}")

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df[args.x_col],
        df[args.y_col],
        c=df[args.color_col],
        cmap="viridis",
        s=8,
        alpha=0.8,
        linewidths=0,
    )
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.title("UBE4 MDS embedding colored by functional score")
    cbar = plt.colorbar(sc)
    cbar.set_label(args.color_col)
    plt.tight_layout()

    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved scatter plot to {output_path}.")


if __name__ == "__main__":
    main()

