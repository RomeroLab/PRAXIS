#!/usr/bin/env python3
"""
activity_axes_trajectory.py

Plot activities (a1, a2, a3) on three axes originating from the center:
- a1 axis points up
- a2 axis points bottom-left
- a3 axis points bottom-right

Rules:
- Group rounds like analysis rules: force first 6 sequences into round 1; then a new round
  starts when created_at exceeds the current round's start time by more than 60 seconds.
- Only consider the first 20 rounds.
- Round 1 (parents): scatter dots (centroids) for all sequences.
- Rounds > 1: for sequences with created_by == 'a2', compute per-round centroid and plot
  a trajectory over time.

Centroid definition:
- Map (a1, a2, a3) to points p1 = a1*v1, p2 = a2*v2, p3 = a3*v3, where
  v1=(0,1), v2=(-sqrt(3)/2,-1/2), v3=(sqrt(3)/2,-1/2). Centroid is (p1+p2+p3)/3
"""

import os
import math
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DB_CANDIDATES = [
    "data/tablespace/self_driving/Duke_final_database.db",
    "data/Duke_final_database.db",
    "data/tablespace/Duke_final_database.db",
    "data/tablespace/training_database.db",
]
MAX_ROUNDS = 20


def find_existing_db_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"SQLite DB not found at explicit path: {explicit_path}")
    for candidate in DEFAULT_DB_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not locate Duke_final_database.db in known locations. Please provide the path.")


def find_table_with_columns(con: sqlite3.Connection) -> Optional[str]:
    needed = {"a1", "a2", "a3", "created_at", "created_by"}
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
    # fallback: accept without created_by (we'll treat as empty, but round 1 scatter will still work)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (tname,) in cur.fetchall():
        try:
            cur.execute(f"PRAGMA table_info({tname})")
            cols = {row[1] for row in cur.fetchall()}
        except Exception:
            continue
        if {"a1", "a2", "a3", "created_at"}.issubset(cols):
            return tname
    return None


def load_from_db(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    db_path = find_existing_db_path(db_path)
    con = sqlite3.connect(db_path)
    try:
        table = find_table_with_columns(con)
        if not table:
            raise RuntimeError("No suitable table with a1,a2,a3,created_at[,created_by] found")
        df = pd.read_sql_query(
            f"SELECT sequence, created_at, created_by, a1, a2, a3 FROM {table}", con
        )
    finally:
        con.close()

    # Parse timestamps and sort
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    # Ensure numeric activities
    for col in ["a1", "a2", "a3"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    # created_by as string
    if "created_by" not in df:
        df["created_by"] = ""
    return df, table


def group_rounds_custom(df: pd.DataFrame) -> pd.DataFrame:
    """Assign rounds: first 6 rows round 1; then cluster by 60-second span."""
    df = df.copy()
    times = df["created_at"].tolist()
    n = len(times)
    if n == 0:
        df["round"] = []
        return df
    rounds = [1] * min(6, n)
    if n > 6:
        current_round = 1
        round_start = times[6]
        for ts in times[6:]:
            if (ts - round_start) > pd.Timedelta(seconds=60):
                current_round += 1
                round_start = ts
            rounds.append(current_round)
    df["round"] = rounds
    return df


def axis_vectors() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1 = np.array([0.0, 1.0])
    v2 = np.array([-math.sqrt(3) / 2.0, -0.5])
    v3 = np.array([ math.sqrt(3) / 2.0, -0.5])
    return v1, v2, v3


def centroid_from_activities(a1: float, a2: float, a3: float) -> Tuple[float, float]:
    v1, v2, v3 = axis_vectors()
    p1 = a1 * v1
    p2 = a2 * v2
    p3 = a3 * v3
    c = (p1 + p2 + p3) / 3.0
    return float(c[0]), float(c[1])


def compute_round_centroid(df_round: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if df_round.empty:
        return None
    # Average activities across sequences in the round
    a1 = df_round["a1"].mean()
    a2 = df_round["a2"].mean()
    a3 = df_round["a3"].mean()
    return centroid_from_activities(a1, a2, a3)


def plot_axes(ax: plt.Axes, max_len: float) -> None:
    v1, v2, v3 = axis_vectors()
    origin = np.array([0.0, 0.0])
    # Draw axes lines
    for v, color in [(v1, "#5a63a4"), (v2, "#fd4470"), (v3, "#fda404")]:
        ax.plot([origin[0], origin[0] + max_len * v[0]], [origin[1], origin[1] + max_len * v[1]],
                color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    # Style
    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.axvline(0, color='lightgray', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    try:
        df, _ = load_from_db(None)
    except Exception as e:
        print(f"Error loading DB: {e}")
        return

    if df.empty:
        print("No data loaded from DB.")
        return

    df = group_rounds_custom(df)
    if MAX_ROUNDS is not None:
        df = df[df["round"] <= MAX_ROUNDS].copy()

    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.suptitle("Activity Trajectory (a2 acquisitions)")

    # Determine axis extent based on activities
    max_activity = float(max(df["a1"].max(), df["a2"].max(), df["a3"].max(), 1.0))
    plot_axes(ax, max_len=max_activity)

    # Round 1: scatter all sequences as centroids
    if (df["round"] == 1).any():
        df_r1 = df[df["round"] == 1]
        pts_x = []
        pts_y = []
        for _, row in df_r1.iterrows():
            x, y = centroid_from_activities(row["a1"], row["a2"], row["a3"])
            pts_x.append(x)
            pts_y.append(y)
        ax.scatter(pts_x, pts_y, s=30, color='#888888', alpha=0.8, label='Round 1 (parents)')

    # Rounds > 1: created_by == 'a2' centroid per round, connect with line
    rounds_sorted = sorted([r for r in df["round"].unique() if r > 1])
    traj_x = []
    traj_y = []
    for r in rounds_sorted:
        created_by_norm = df["created_by"].astype(str).str.lower()
        sub = df[(df["round"] == r) & (created_by_norm == 'agent_a2')]
        c = compute_round_centroid(sub)
        if c is not None:
            traj_x.append(c[0])
            traj_y.append(c[1])
    if traj_x:
        ax.plot(traj_x, traj_y, '-', color='#5a63a4', linewidth=3.0, alpha=0.9, label='Trajectory (a2)')

    # Labels and styling
    ax.legend(loc='upper right', frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    out_path = "data_analysis/activity_axes_trajectory.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved activity trajectory plot to {out_path}")


if __name__ == "__main__":
    main()


