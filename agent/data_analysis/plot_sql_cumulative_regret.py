#!/usr/bin/env python3
"""
plot_sql_cumulative_regret.py

Generates two CSVs used by analyze_ucb_regret_behavior.py from the SQLite DB:
- sql_round_specificity_summary.csv: per-round best experimental specificities (spec1/spec2/spec3)
- sql_cumulative_regret_pivot.csv: cumulative max of those best specificities by round

Notes:
- The first 6 sequences (after sorting by created_at) are forced into round 1.
- Remaining sequences are clustered into rounds; a new round starts when created_at
  exceeds the current round's start time by more than 60 seconds.
- Specificity formula matches self_driving.calculate_enzyme_specificity without sigma terms:
  spec(target, off1, off2) = (target^2) / (target + off1 + off2)   [L1-based]
"""

import os
import sqlite3
import pandas as pd
import numpy as np


def find_db_and_table():
    """Return (db_path, table_name) for a DB containing a1,a2,a3,created_at."""
    candidates = [
        "data/tablespace/self_driving/Duke_final_database.db",
        "data/Duke_final_database.db",
        "insilico_analysis/ube4/data/tablespace/ube4_synevo/ube4_final_database.db",
        "insilico_analysis/gfp/GFP_final_database.db",
    ]
    for db in candidates:
        if not os.path.exists(db):
            continue
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for (tname,) in cur.fetchall():
                try:
                    cur.execute(f"PRAGMA table_info({tname})")
                    cols = [r[1] for r in cur.fetchall()]
                except Exception:
                    continue
                needed = {"a1", "a2", "a3", "created_at"}
                if needed.issubset(set(cols)):
                    con.close()
                    return db, tname
            con.close()
        except Exception:
            pass
    return None, None


def compute_specificities(df: pd.DataFrame) -> pd.DataFrame:
    """Add experimental specificities per substrate to DataFrame (no sigma)."""
    epsilon = 1e-12
    for col in ["a1", "a2", "a3"]:
        if col not in df.columns:
            df[col] = np.nan
    # Ensure numeric
    for col in ["a1", "a2", "a3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    def spec(t, o1, o2):
        denom = t + o1 + o2 + epsilon
        return (t ** 2) / denom

    df["exp_spec_s1"] = spec(df["a1"].to_numpy(float), df["a2"].to_numpy(float), df["a3"].to_numpy(float))
    df["exp_spec_s2"] = spec(df["a2"].to_numpy(float), df["a1"].to_numpy(float), df["a3"].to_numpy(float))
    df["exp_spec_s3"] = spec(df["a3"].to_numpy(float), df["a1"].to_numpy(float), df["a2"].to_numpy(float))
    return df


def main():
    db_path, table_name = find_db_and_table()
    if not db_path:
        print("Error: Could not locate a SQLite DB with required columns a1,a2,a3,created_at")
        return

    print(f"Using DB: {db_path}\nTable: {table_name}")
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT sequence, created_at, a1, a2, a3 FROM {table_name}", con)
    finally:
        con.close()

    # Parse timestamps and sort
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    if df.empty:
        print("Error: No rows with valid created_at found in the database table.")
        return

    # Compute experimental specificities
    df = compute_specificities(df)

    # Infer rounds with custom rule: first 6 rows are round 1; then cluster rest with 60s threshold
    times = df["created_at"].tolist()
    if not times:
        print("Error: No valid timestamps after sorting.")
        return
    n = len(times)
    rounds = [1] * min(6, n)
    if n > 6:
        current_round = 1
        round_start = times[0]  # round 1 start at first timestamp
        # For subsequent items, allow new rounds when >60s from last round_start
        round_start = times[6]  # start timing for the post-seed cluster at item 6
        for ts in times[6:]:
            if (ts - round_start) > pd.Timedelta(seconds=60):
                current_round += 1
                round_start = ts
            rounds.append(current_round)
    df["round"] = rounds

    # Per-round best specificities (new in that round only)
    round_grp = df.groupby("round", as_index=False)
    round_summary = round_grp.agg(
        best_spec1=("exp_spec_s1", "max"),
        best_spec2=("exp_spec_s2", "max"),
        best_spec3=("exp_spec_s3", "max"),
    ).sort_values("round").reset_index(drop=True)

    # Save per-round summary
    round_summary.to_csv("sql_round_specificity_summary.csv", index=False)
    print("Wrote sql_round_specificity_summary.csv")

    # Cumulative max pivot (columns: round, spec1, spec2, spec3)
    cum = round_summary.copy()
    cum["spec1"] = cum["best_spec1"].cummax()
    cum["spec2"] = cum["best_spec2"].cummax()
    cum["spec3"] = cum["best_spec3"].cummax()
    pivot = cum[["round", "spec1", "spec2", "spec3"]]
    pivot.to_csv("sql_cumulative_regret_pivot.csv", index=False)
    print("Wrote sql_cumulative_regret_pivot.csv")


if __name__ == "__main__":
    main()


