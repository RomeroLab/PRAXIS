#!/usr/bin/env python3
"""Plot the gfp MDS embedding as a scatter plot colored by functional score."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MDS scatter plot for gfp.")
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(
            "insilico_analysis/dataset_oracle/gfp/seq_vis/gfp_mds_embedding.csv"
        ),
        help="CSV produced by run_mds.py containing sequence coordinates and scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "insilico_analysis/dataset_oracle/gfp/seq_vis/gfp_mds_scatter.png"
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
    parser.add_argument(
        "--highlight-db",
        type=Path,
        default=Path(
            "insilico_analysis/dataset_oracle/gfp/data/tablespace/gfp_frontier/GFP_poisson_all_final_database.db"
        ),
        help="SQLite database containing sequences to highlight.",
    )
    parser.add_argument(
        "--highlight-limit",
        type=int,
        default=200,
        help="Number of sequences to fetch for highlighting.",
    )
    parser.add_argument(
        "--highlight-color",
        default="#fd4470",
        help="Color used to highlight selected sequences.",
    )
    return parser.parse_args()


def _normalize_sequence(seq: str) -> str:
    """Return a canonical representation for sequence matching."""

    return str(seq).replace("*", "").replace("-", "").strip()


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

    highlight_sequences: set[str] = set()
    highlight_df = pd.DataFrame(columns=df.columns)
    if args.highlight_db is not None:
        highlight_path = args.highlight_db.expanduser().resolve()
        if highlight_path.exists():
            with sqlite3.connect(highlight_path) as con:
                query = (
                    "SELECT sequence FROM GFP_poisson_all_final_assayed_sequences "
                    "ORDER BY sequence_id LIMIT ?"
                )
                fetched = pd.read_sql_query(query, con, params=(args.highlight_limit,))
                highlight_sequences = {
                    _normalize_sequence(seq)
                    for seq in fetched["sequence"].astype(str)
                }
                highlight_df = df[
                    df["sequence"].astype(str).map(_normalize_sequence).isin(highlight_sequences)
                ]
        else:
            print(f"Warning: highlight database not found at {highlight_path}; skipping highlights.")

    if args.sample is not None and args.sample < len(df):
        rng = args.seed if args.seed is not None else None
        non_highlight = df[~df["sequence"].astype(str).map(_normalize_sequence).isin(highlight_sequences)]
        sampled = non_highlight.sample(n=min(args.sample, len(non_highlight)), random_state=rng)
        df = pd.concat([sampled, highlight_df], ignore_index=True)
        if "sequence" in df.columns:
            df = df.drop_duplicates(subset=["sequence"])
    else:
        df = df.copy()

    mask = df["sequence"].astype(str).map(_normalize_sequence).isin(highlight_sequences)
    non_highlighted = df.loc[~mask]
    highlighted = df.loc[mask]

    plt.figure(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list(
        "functional_score",
        [(0.0, "#d3d3d3"), (0.7, "#d3d3d3"), (1.0, "#05a083")],
    )
    sc = plt.scatter(
        non_highlighted[args.x_col],
        non_highlighted[args.y_col],
        c=non_highlighted[args.color_col],
        cmap=cmap,
        s=8,
        alpha=0.8,
        linewidths=0,
    )
    if not highlighted.empty:
        plt.scatter(
            highlighted[args.x_col],
            highlighted[args.y_col],
            c=args.highlight_color,
            s=8,
            alpha=0.9,
            linewidths=0,
            label="Highlighted sequences",
        )
        plt.legend(loc="upper right", frameon=False)
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.title("gfp MDS embedding colored by functional score")
    cbar = plt.colorbar(sc)
    cbar.set_label(args.color_col)
    plt.tight_layout()

    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved scatter plot to {output_path}.")


if __name__ == "__main__":
    main()

