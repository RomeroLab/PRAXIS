"""
Metrics for in-silico benchmarking of the self-driving BO pipeline.

All functions operate on a round-log DataFrame (columns: round, sequence, <phenotype>)
and optionally the full dataset for top-K calculations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def compute_topk_discovery_rate(
    round_log_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    phenotype_col: str,
    seq_col: str = "sequence",
    k_values: List[int] = [10, 50, 100],
) -> pd.DataFrame:
    """
    At each round, compute fraction of true top-K sequences discovered so far.

    Returns DataFrame with columns: round, k, fraction_discovered
    """
    # Normalize sequences
    dataset_df = dataset_df.copy()
    dataset_df[seq_col] = dataset_df[seq_col].astype(str).str.replace("*", "", regex=False)
    round_log_df = round_log_df.copy()
    round_log_df["sequence"] = round_log_df["sequence"].astype(str).str.replace("*", "", regex=False)

    rows = []
    rounds = sorted(round_log_df["round"].unique())
    for k in k_values:
        # Identify true top-K sequences from full dataset
        top_k_seqs = set(dataset_df.nlargest(k, phenotype_col)[seq_col].tolist())
        discovered = set()
        for r in rounds:
            round_seqs = set(round_log_df[round_log_df["round"] == r]["sequence"].tolist())
            discovered |= round_seqs
            fraction = len(discovered & top_k_seqs) / k
            rows.append({"round": r, "k": k, "fraction_discovered": fraction})
    return pd.DataFrame(rows)


def compute_time_to_threshold(
    round_log_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    phenotype_col: str,
    seq_col: str = "sequence",
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99],
) -> Dict[float, float]:
    """
    Cumulative max fitness vs. global max. Returns dict: {threshold: num_queries_or_nan}.
    """
    dataset_df = dataset_df.copy()
    dataset_df[seq_col] = dataset_df[seq_col].astype(str).str.replace("*", "", regex=False)
    round_log_df = round_log_df.copy()
    round_log_df["sequence"] = round_log_df["sequence"].astype(str).str.replace("*", "", regex=False)

    global_max = dataset_df[phenotype_col].max()
    if global_max == 0:
        return {t: float("nan") for t in thresholds}

    # Sort by round, compute cumulative max
    rl = round_log_df.sort_values("round").reset_index(drop=True)
    rl["cummax"] = rl[phenotype_col].cummax()
    rl["fraction_of_max"] = rl["cummax"] / global_max

    result = {}
    for t in thresholds:
        hit = rl[rl["fraction_of_max"] >= t]
        if len(hit) > 0:
            # num_queries = index of first hit + 1 (1-indexed count)
            result[t] = int(hit.index[0]) + 1
        else:
            result[t] = float("nan")
    return result


def compute_sample_efficiency(
    round_log_df: pd.DataFrame,
    phenotype_col: str,
) -> pd.DataFrame:
    """
    Cumulative max fitness at each query.

    Returns DataFrame with columns: num_queries, best_fitness
    """
    rl = round_log_df.sort_values("round").reset_index(drop=True)
    rl["cummax"] = rl[phenotype_col].cummax()
    rows = []
    for idx in range(len(rl)):
        rows.append({
            "num_queries": idx + 1,
            "best_fitness": rl.loc[idx, "cummax"],
        })
    return pd.DataFrame(rows)
