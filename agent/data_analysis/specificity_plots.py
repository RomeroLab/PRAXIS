#!/usr/bin/env python3
"""
specificity_plots.py

Reads SQLite data containing activities a1, a2, a3 and timestamps, groups
sequences into rounds by rounding created_at to the nearest minute, computes
per-substrate specificities and plots per-round best vs cumulative best for
each substrate on three subplots.

Specificity definition (per substrate i), L1-based:
  a_i^2 / (a1 + a2 + a3)  =  a_i * (a_i / total_activity)

Outputs (limited to first 20 rounds by default):
- sql_round_specificity_summary.csv: per-round best specificities (spec1..3)
- sql_cumulative_regret_pivot.csv: cumulative max of best specificities by round
- data_analysis/specificity_plots.png: figure with three subplots
"""

import os
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DB_CANDIDATES = [
    "data/tablespace/self_driving/Duke_final_database.db",
    "data/Duke_final_database.db",
    "data/tablespace/Duke_final_database.db",
    "data/tablespace/training_database.db",  # fallback
]
MAX_ROUNDS = 20


def find_existing_db_path(explicit_path: Optional[str] = None) -> str:
    """Return a valid DB path. Prefer explicit path; otherwise search candidates."""
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"SQLite DB not found at explicit path: {explicit_path}")
    for candidate in DEFAULT_DB_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate Duke_final_database.db in known locations. "
        "Please provide the path."
    )


def find_table_with_columns(con: sqlite3.Connection) -> Optional[str]:
    """Return a table name in the open SQLite connection that has needed columns.

    Needed columns: a1, a2, a3, created_at
    """
    needed = {"a1", "a2", "a3", "created_at"}
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (tname,) in cur.fetchall():
        try:
            cur.execute(f"PRAGMA table_info({tname})")
            cols = {row[1] for row in cur.fetchall()}
        except Exception:
            continue
        if needed.issubset(cols):
            return tname
    return None


def load_activities_from_db(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load a DataFrame with columns sequence, created_at, a1, a2, a3 from DB.

    Auto-detect the table that contains required columns.
    """
    db_path = find_existing_db_path(db_path)

    con = sqlite3.connect(db_path)
    try:
        table_name = find_table_with_columns(con)
        if not table_name:
            raise RuntimeError("No table with columns a1,a2,a3,created_at found in DB")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    finally:
        con.close()

    # Parse timestamps and sort
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    return df, table_name


def compute_specificities(df: pd.DataFrame) -> pd.DataFrame:
    """Compute experimental specificities for each substrate.

    spec_i = a_i^2 / (a1 + a2 + a3 + eps)   [L1-based]
    """
    epsilon = 1e-12
    for col in ["a1", "a2", "a3"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)

    denom = df["a1"] + df["a2"] + df["a3"] + epsilon
    df["exp_spec_s1"] = (df["a1"] ** 2) / denom
    df["exp_spec_s2"] = (df["a2"] ** 2) / denom
    df["exp_spec_s3"] = (df["a3"] ** 2) / denom
    return df


def group_rounds_by_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'round' column by rounding created_at to nearest minute and ordering.

    Rounds are 1..N in ascending order of the rounded-minute timestamp.
    """
    df = df.copy()
    rounded = df["created_at"].dt.round("T")  # nearest minute
    df["rounded_minute"] = rounded
    unique_minutes = (
        df[["rounded_minute"]]
        .drop_duplicates()
        .sort_values("rounded_minute")
        .reset_index(drop=True)
    )
    unique_minutes["round"] = np.arange(1, len(unique_minutes) + 1)
    df = df.merge(unique_minutes, on="rounded_minute", how="left")
    df["round"] = df["round"].astype(int)
    return df


def summarize_per_round(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-round best specificities for each substrate."""
    grp = df.groupby("round", as_index=False)
    round_summary = grp.agg(
        best_spec1=("exp_spec_s1", "max"),
        best_spec2=("exp_spec_s2", "max"),
        best_spec3=("exp_spec_s3", "max"),
    ).sort_values("round").reset_index(drop=True)
    return round_summary


def cumulative_pivot(round_summary: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative max pivot matching prior analysis file shapes."""
    cum = round_summary.copy()
    cum["spec1"] = cum["best_spec1"].cummax()
    cum["spec2"] = cum["best_spec2"].cummax()
    cum["spec3"] = cum["best_spec3"].cummax()
    pivot = cum[["round", "spec1", "spec2", "spec3"]]
    return pivot


def plot_specificities(round_summary: pd.DataFrame) -> None:
    """Plot per-round best and cumulative best for each substrate on 3 subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("UCB Analysis: Cumulative Max with Per-Round Specificity (SQL)", fontsize=16)

    # Style axes to match analyze_ucb_regret_behavior.py
    for j in range(3):
        axes[j].set_facecolor('#ffffff')
        for spine in axes[j].spines.values():
            spine.set_linewidth(3.5)
        axes[j].tick_params(which='both', width=2, labelsize=12, length=8)
        axes[j].minorticks_off()

    colors = ["#5a63a4", "#fd4470", "#fda404"]
    titles = ["A1", "A2", "A3"]
    rounds = round_summary["round"].tolist()

    for idx, (spec_col, title) in enumerate(
        [("best_spec1", titles[0]), ("best_spec2", titles[1]), ("best_spec3", titles[2])]
    ):
        ax = axes[idx]
        per_round = round_summary[spec_col].tolist()
        cum_best = pd.Series(per_round).cummax().tolist()

        # Per-round best (underlay)
        ax.plot(rounds, per_round, '-', color=colors[idx], linewidth=3, alpha=0.25)
        # Cumulative best (overlay)
        ax.plot(rounds, cum_best, '-', color=colors[idx], linewidth=5, alpha=0.9)

        # Final max line
        if cum_best:
            ax.axhline(y=cum_best[-1], color=colors[idx], linestyle='--', alpha=0.4, linewidth=3)

        ax.set_xlabel('Round')
        ax.set_ylabel('Specificity')
        ax.set_title(f"{title}: Cumulative vs Per-Round")

        if rounds:
            max_round = max(rounds)
            ax.set_xticks(range(2, max_round + 1, 2))

        # Vertical lines per round
        for r in rounds:
            ax.axvline(x=r, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5)

        # No legend as per reference formatting

    # Final styling for ticks
    for j in range(3):
        for label in axes[j].get_xticklabels() + axes[j].get_yticklabels():
            label.set_fontweight('normal')
            label.set_fontfamily('sans-serif')

    plt.tight_layout()
    out_path = "data_analysis/specificity_plots.png"
    plt.savefig(out_path, dpi=800, bbox_inches='tight')
    plt.close(fig)


def print_round_best_details(df: pd.DataFrame) -> None:
    """Print, for each plot (A1/A2/A3), the per-round best and cumulative best series."""
    if df.empty or "round" not in df.columns:
        print("No round data to summarize.")
        return

    round_summary = summarize_per_round(df)
    titles = ["A1", "A2", "A3"]
    cols = ["best_spec1", "best_spec2", "best_spec3"]

    for title, col in zip(titles, cols):
        per_round = round_summary[col].tolist()
        cum_best = pd.Series(per_round).cummax().tolist()
        print(f"{title}:")
        print(f"  per_round={per_round}")
        print(f"  cumulative_best={cum_best}")


def export_with_specificities(df: pd.DataFrame, db_path: Optional[str] = None) -> str:
    """Export all original DB columns plus spec_1, spec_2, spec_3 to a CSV.

    Specificities are computed as:
      spec_i = a_i^2 / (a1 + a2 + a3 + eps)   [L1-based]

    Returns the output CSV path.
    """
    out = df.copy()

    epsilon = 1e-12
    for col in ["a1", "a2", "a3"]:
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)

    denom = out["a1"] + out["a2"] + out["a3"] + epsilon
    out["spec_glu"] = (out["a1"] ** 2) / denom
    out["spec_xyl"] = (out["a2"] ** 2) / denom
    out["spec_man"] = (out["a3"] ** 2) / denom

    # Rename activity columns to substrate names
    out = out.rename(columns={"a1": "a_glu", "a2": "a_xyl", "a3": "a_man"})

    # Drop intermediate and unwanted columns
    cols_to_drop = [
        c for c in ["sequence_id", "mutant",
                     "exp_spec_s1", "exp_spec_s2", "exp_spec_s3",
                     "rounded_minute", "round"]
        if c in out.columns
    ]
    out = out.drop(columns=cols_to_drop)

    out_path = "data_analysis/specificity_with_activities.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    try:
        df, table = load_activities_from_db(None)
    except Exception as e:
        print(f"Error loading DB: {e}")
        return

    if df.empty:
        print("No data loaded from DB.")
        return

    # Export full DB rows + spec_1, spec_2, spec_3 before any filtering
    spec_csv_path = export_with_specificities(df)
    print(f"Wrote {spec_csv_path}")

    df = compute_specificities(df)
    df = group_rounds_by_minute(df)
    # Limit to first MAX_ROUNDS
    if MAX_ROUNDS is not None:
        df = df[df["round"] <= MAX_ROUNDS].copy()

    round_summary = summarize_per_round(df)
    round_summary.to_csv("sql_round_specificity_summary.csv", index=False)

    pivot = cumulative_pivot(round_summary)
    pivot.to_csv("sql_cumulative_regret_pivot.csv", index=False)

    plot_specificities(round_summary)

    # Print per-round best details for traceability
    print_round_best_details(df)

    # Minimal console feedback
    print("Wrote sql_round_specificity_summary.csv and sql_cumulative_regret_pivot.csv")
    print("Saved figure to data_analysis/specificity_plots.png")


if __name__ == "__main__":
    main()


