#!/usr/bin/env python3
"""
spec2_cumulative_centroid.py

Plot the trajectory of centroids for the most a2-specific sequence in each round
on three axes from the origin:

Rounds:
- First 6 sequences (by created_at) are forced into round 1
- Subsequent rounds start when created_at advances >60s from current round start
- Limited to first 20 rounds

For each round r:
- Identify the sequence within the cumulative data up to r with the highest exp_spec_s2
- Map its activities (a1,a2,a3) to points p1=a1*v1, p2=a2*v2, p3=a3*v3, and take
  centroid c=(p1+p2+p3)/3. Connect these centroids with a continuous line.
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

# =====================
# Simple toggles to show/hide plot elements (easy to comment out)
# =====================
SHOW = {
    'spec1_cumulative': False,
    'spec1_per_round': False,
    'spec2_cumulative': True,
    'spec2_per_round': True,
    'spec3_cumulative': False,
    'spec3_per_round': False,
    'round1_black_dots': True,
    'spec1_increase_markers': True,
    'spec2_increase_markers': True,
    'spec3_increase_markers': True,
}

COLORS = {
    'spec1': '#5a63a4',  # blue
    'spec2': '#fd4470',  # red
    'spec3': '#fda404',  # yellow
}


def find_existing_db_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"SQLite DB not found at explicit path: {explicit_path}")
    for cand in DEFAULT_DB_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("Could not locate Duke_final_database.db; provide a path.")


def find_table_with_required_columns(con: sqlite3.Connection) -> Optional[str]:
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


def load_activities(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    db_path = find_existing_db_path(db_path)
    con = sqlite3.connect(db_path)
    try:
        table = find_table_with_required_columns(con)
        if not table:
            raise RuntimeError("No table with columns a1,a2,a3,created_at found.")
        df = pd.read_sql_query(
            f"SELECT sequence, created_at, a1, a2, a3 FROM {table}", con
        )
    finally:
        con.close()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    for col in ["a1", "a2", "a3"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    return df, table


def compute_specificities(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-12
    denom = df["a1"] + df["a2"] + df["a3"] + epsilon
    df["exp_spec_s1"] = (df["a1"] ** 2) / denom
    df["exp_spec_s2"] = (df["a2"] ** 2) / denom
    df["exp_spec_s3"] = (df["a3"] ** 2) / denom
    return df


def group_rounds_custom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    times = df["created_at"].tolist()
    n = len(times)
    if n == 0:
        df["round"] = []
        return df
    # Exactly first 6 sequences are round 1
    rounds = [1] * min(6, n)
    if n > 6:
        # Start clustering from sequence index 6 as round 2
        current_round = 2
        round_start = times[6]
        for ts in times[6:]:
            if (ts - round_start) > pd.Timedelta(seconds=60):
                current_round += 1
                round_start = ts
            rounds.append(current_round)
    df["round"] = rounds
    return df


def axis_vectors():
    v1 = np.array([0.0, 1.0])
    v2 = np.array([-math.sqrt(3) / 2.0, -0.5])
    v3 = np.array([ math.sqrt(3) / 2.0, -0.5])
    return v1, v2, v3


def threshold_activities(a1: float, a2: float, a3: float, threshold: float = 1e-4) -> Tuple[float, float, float]:
    """Set activity values below threshold to zero."""
    a1t = a1 if a1 >= threshold else 0.0
    a2t = a2 if a2 >= threshold else 0.0
    a3t = a3 if a3 >= threshold else 0.0
    return a1t, a2t, a3t


def plot_axes(ax: plt.Axes, max_len: float) -> None:
    v1, v2, v3 = axis_vectors()
    origin = np.array([0.0, 0.0])
    # Draw axes in black
    for v in (v1, v2, v3):
        ax.plot([origin[0], origin[0] + max_len * v[0]], [origin[1], origin[1] + max_len * v[1]],
                color='black', linestyle='-', linewidth=3.0, alpha=0.9, zorder=1)
    # Add tick marks along the three radial axes (not Cartesian axes)
    import numpy as _np
    tick_vals = _np.linspace(max_len/5.0, max_len, 5)  # 5 ticks per axis
    tick_len = 0.06 * max_len
    label_offset = 0.025 * max_len
    for v in (v1, v2, v3):
        # Perpendicular unit vector for tick orientation
        n = _np.array([-v[1], v[0]])
        # v and n are already unit-length for our chosen v1,v2,v3
        for t in tick_vals:
            c = origin + t * v
            p1 = c - (tick_len/2.0) * n
            p2 = c + (tick_len/2.0) * n
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=2.0, zorder=1)
            # Label just beyond the tick in the +n direction
            lp = p2 + label_offset * n
            ax.text(lp[0], lp[1], f"{t:.1f}", fontsize=9, color='black', ha='center', va='center', zorder=1)
    ax.set_aspect('equal', adjustable='box')
    # Hide default Cartesian ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Set limits so the top of a1 is near top, include all axes with small margin
    endpoints = np.stack([
        origin + max_len * v1,
        origin + max_len * v2,
        origin + max_len * v3,
        origin
    ], axis=0)
    margin = 0.05 * max_len
    x_min, y_min = endpoints.min(axis=0) - margin
    x_max, y_max = endpoints.max(axis=0) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def main() -> None:
    try:
        df, _ = load_activities(None)
    except Exception as e:
        print(f"Error loading DB: {e}")
        return

    if df.empty:
        print("No data loaded from DB.")
        return

    df = compute_specificities(df)
    df = group_rounds_custom(df)
    if MAX_ROUNDS is not None:
        df = df[df["round"] <= MAX_ROUNDS].copy()

    # For round 1: use centroid of all parent sequences (expected high a1)
    # For each round r>=2: find the cumulatively best spec2 among rounds <= r, then take its centroid from (a1,a2,a3)
    v1, v2, v3 = axis_vectors()
    traj = []
    traj1 = []
    traj3 = []
    rounds = sorted(df["round"].unique())
    best_idx_so_far = None
    best_spec2_so_far = -1.0
    spec2_increase_points = []  # (round, x, y)
    for r in rounds:
        sub_cum = df[df["round"] <= r]
        # Cumulative best spec2 up to r (for r==1 this is the best parent on spec2)
        idx2 = sub_cum["exp_spec_s2"].idxmax()
        spec2_val = float(sub_cum.loc[idx2, "exp_spec_s2"]) if idx2 is not None else -1.0
        if spec2_val >= best_spec2_so_far:
            best_spec2_so_far = spec2_val
            best_idx_so_far = idx2
        if best_idx_so_far is not None:
            row2 = df.loc[best_idx_so_far]
            a1v = float(row2["a1"])
            a2v = float(row2["a2"])
            a3v = float(row2["a3"])
            a1t, a2t, a3t = threshold_activities(a1v, a2v, a3v)
            c = (a1t * v1 + a2t * v2 + a3t * v3) / 3.0
            traj.append((float(c[0]), float(c[1])))
            # increase markers computed from per-round best list later

        # Cumulative best spec1 up to r
        idx1 = sub_cum["exp_spec_s1"].idxmax()
        if idx1 is not None:
            row1 = df.loc[idx1]
            d1 = float(row1["a1"]); d2 = float(row1["a2"]); d3 = float(row1["a3"])
            d1t, d2t, d3t = threshold_activities(d1, d2, d3)
            c1 = (d1t * v1 + d2t * v2 + d3t * v3) / 3.0
            traj1.append((float(c1[0]), float(c1[1])))

        # Cumulative best spec3 up to r
        sub_cum3 = sub_cum
        idx3 = sub_cum3["exp_spec_s3"].idxmax()
        if idx3 is not None:
            row3 = df.loc[idx3]
            b1v = float(row3["a1"])
            b2v = float(row3["a2"])
            b3v = float(row3["a3"])
            b1t, b2t, b3t = threshold_activities(b1v, b2v, b3v)
            c3 = (b1t * v1 + b2t * v2 + b3t * v3) / 3.0
            traj3.append((float(c3[0]), float(c3[1])))

    traj = np.array(traj, dtype=float) if len(traj) > 0 else np.zeros((0,2), dtype=float)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.suptitle("Cumulative Best Spec2: Activity Centroid Trajectory")
    # We'll draw trajectory first, then axes, then explicitly lock limits
    axes_len = 2.5

    # Convert cumulative spec1 and spec3 trajectories to numpy array
    traj1 = np.array(traj1, dtype=float) if len(traj1) > 0 else np.zeros((0,2), dtype=float)
    traj3 = np.array(traj3, dtype=float) if len(traj3) > 0 else np.zeros((0,2), dtype=float)

    # Also compute per-round best spec2 centroid trajectory (dimmer red)
    per_round_traj = []
    per_round_best_spec2_vals = []
    for r in rounds:
        sub_r = df[df["round"] == r]
        if sub_r.empty:
            continue
        idx_r = sub_r["exp_spec_s2"].idxmax()
        row_r = df.loc[idx_r]
        a1r, a2r, a3r = threshold_activities(float(row_r["a1"]), float(row_r["a2"]), float(row_r["a3"]))
        c_r = (a1r * v1 + a2r * v2 + a3r * v3) / 3.0
        per_round_traj.append((float(c_r[0]), float(c_r[1])))
        per_round_best_spec2_vals.append(float(sub_r.loc[idx_r, "exp_spec_s2"]))
    per_round_traj = np.array(per_round_traj, dtype=float) if len(per_round_traj) > 0 else np.zeros((0,2), dtype=float)

    # Compute spec2 cumulative increase markers like specificity_plots (based on per-round best series)
    if SHOW['spec2_increase_markers'] and len(per_round_best_spec2_vals) > 0:
        cum = -1.0
        inc_coords = []
        inc_rounds = []
        eps = 1e-12
        for i, val in enumerate(per_round_best_spec2_vals):
            if val > cum + eps:
                cum = val
                inc_coords.append(per_round_traj[i])
                inc_rounds.append(int(rounds[i]))
        spec2_increase_points = [(inc_rounds[i], float(inc_coords[i][0]), float(inc_coords[i][1])) for i in range(len(inc_rounds))]

    # Per-round best spec1 centroid trajectory (dimmer blue)
    per_round_traj1 = []
    per_round_best_spec1_vals = []
    for r in rounds:
        sub_r = df[df["round"] == r]
        if sub_r.empty:
            continue
        idx_r1 = sub_r["exp_spec_s1"].idxmax()
        row_r1 = df.loc[idx_r1]
        e1, e2, e3 = threshold_activities(float(row_r1["a1"]), float(row_r1["a2"]), float(row_r1["a3"]))
        c_1 = (e1 * v1 + e2 * v2 + e3 * v3) / 3.0
        per_round_traj1.append((float(c_1[0]), float(c_1[1])))
        per_round_best_spec1_vals.append(float(sub_r.loc[idx_r1, "exp_spec_s1"]))
    per_round_traj1 = np.array(per_round_traj1, dtype=float) if len(per_round_traj1) > 0 else np.zeros((0,2), dtype=float)

    # Per-round best spec3 centroid trajectory (dimmer yellow)
    per_round_traj3 = []
    per_round_best_spec3_vals = []
    for r in rounds:
        sub_r = df[df["round"] == r]
        if sub_r.empty:
            continue
        idx_r3 = sub_r["exp_spec_s3"].idxmax()
        row_r3 = df.loc[idx_r3]
        c1, c2, c3v = threshold_activities(float(row_r3["a1"]), float(row_r3["a2"]), float(row_r3["a3"]))
        c_3 = (c1 * v1 + c2 * v2 + c3v * v3) / 3.0
        per_round_traj3.append((float(c_3[0]), float(c_3[1])))
        per_round_best_spec3_vals.append(float(sub_r.loc[idx_r3, "exp_spec_s3"]))
    per_round_traj3 = np.array(per_round_traj3, dtype=float) if len(per_round_traj3) > 0 else np.zeros((0,2), dtype=float)

    # spec2 cumulative (red)
    if SHOW['spec2_cumulative']:
        if len(traj) >= 2:
            ax.plot(traj[:, 0], traj[:, 1], '-', color=COLORS['spec2'], linewidth=5.0, alpha=0.9, zorder=3)
        if len(traj) >= 1:
            ax.scatter(traj[0, 0], traj[0, 1], s=64, color=COLORS['spec2'], alpha=1, zorder=4)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=112, color=COLORS['spec2'], edgecolors='white', linewidths=0.8, alpha=1.0, zorder=5)
        # Mark spec2 cumulative increase points
        if SHOW['spec2_increase_markers'] and spec2_increase_points:
            inc = np.array([(x, y) for (_, x, y) in spec2_increase_points], dtype=float)
            ax.scatter(inc[:,0], inc[:,1], s=80, facecolors='none', edgecolors=COLORS['spec2'], linewidths=1.25, zorder=4.5)
            print("Spec2 cumulative increases at rounds:", [int(r) for (r,_,_) in spec2_increase_points])

    # Plot per-round best dimmer red trajectory
    # spec2 per-round (dimmer red)
    if SHOW['spec2_per_round'] and len(per_round_traj) >= 2:
        ax.plot(per_round_traj[:, 0], per_round_traj[:, 1], '-', color=COLORS['spec2'], linewidth=3.0, alpha=0.25, zorder=2.6)

    # Plot cumulative best spec1 (blue) trajectory and endpoints
    # spec1 cumulative (blue)
    if SHOW['spec1_cumulative']:
        if len(traj1) >= 2:
            ax.plot(traj1[:, 0], traj1[:, 1], '-', color=COLORS['spec1'], linewidth=5.0, alpha=0.9, zorder=3)
        if len(traj1) >= 1:
            ax.scatter(traj1[0, 0], traj1[0, 1], s=64, color=COLORS['spec1'], alpha=1, zorder=4)
            ax.scatter(traj1[-1, 0], traj1[-1, 1], s=112, color=COLORS['spec1'], edgecolors='white', linewidths=0.8, alpha=1.0, zorder=5)

    # Plot per-round best dimmer blue trajectory
    # spec1 per-round (dimmer blue)
    if SHOW['spec1_per_round'] and len(per_round_traj1) >= 2:
        ax.plot(per_round_traj1[:, 0], per_round_traj1[:, 1], '-', color=COLORS['spec1'], linewidth=3.0, alpha=0.25, zorder=2.6)
    # Mark spec1 cumulative increase points (from per-round best series)
    if SHOW['spec1_increase_markers'] and len(per_round_best_spec1_vals) > 0:
        cum = -1.0
        inc_coords = []
        inc_rounds = []
        eps = 1e-12
        for i, val in enumerate(per_round_best_spec1_vals):
            if val > cum + eps:
                cum = val
                inc_coords.append(per_round_traj1[i])
                inc_rounds.append(int(rounds[i]))
        if inc_coords:
            inc = np.array(inc_coords, dtype=float)
            ax.scatter(inc[:,0], inc[:,1], s=80, facecolors='none', edgecolors=COLORS['spec1'], linewidths=1.25, zorder=4.5)
            print("Spec1 cumulative increases at rounds:", inc_rounds)

    # Plot cumulative best spec3 (yellow) trajectory and endpoints
    # spec3 cumulative (yellow)
    if SHOW['spec3_cumulative']:
        if len(traj3) >= 2:
            ax.plot(traj3[:, 0], traj3[:, 1], '-', color=COLORS['spec3'], linewidth=5.0, alpha=0.9, zorder=3)
        if len(traj3) >= 1:
            ax.scatter(traj3[0, 0], traj3[0, 1], s=64, color=COLORS['spec3'], alpha=1, zorder=4)
            ax.scatter(traj3[-1, 0], traj3[-1, 1], s=112, color=COLORS['spec3'], edgecolors='white', linewidths=0.8, alpha=1.0, zorder=5)

    # Plot per-round best dimmer yellow trajectory
    # spec3 per-round (dimmer yellow)
    if SHOW['spec3_per_round'] and len(per_round_traj3) >= 2:
        ax.plot(per_round_traj3[:, 0], per_round_traj3[:, 1], '-', color=COLORS['spec3'], linewidth=3.0, alpha=0.25, zorder=2.6)
    # Mark spec3 cumulative increase points (from per-round best series)
    if SHOW['spec3_increase_markers'] and len(per_round_best_spec3_vals) > 0:
        cum = -1.0
        inc_coords = []
        inc_rounds = []
        eps = 1e-12
        for i, val in enumerate(per_round_best_spec3_vals):
            if val > cum + eps:
                cum = val
                inc_coords.append(per_round_traj3[i])
                inc_rounds.append(int(rounds[i]))
        if inc_coords:
            inc = np.array(inc_coords, dtype=float)
            ax.scatter(inc[:,0], inc[:,1], s=80, facecolors='none', edgecolors=COLORS['spec3'], linewidths=1.25, zorder=4.5)
            print("Spec3 cumulative increases at rounds:", inc_rounds)

    # Scatter centroids of all individual sequences in round 1 as black dots
    r1 = df[df["round"] == 1]
    if SHOW['round1_black_dots'] and not r1.empty:
        v1, v2, v3 = axis_vectors()
        xs = []
        ys = []
        for _, row in r1.iterrows():
            a1v = float(row["a1"])
            a2v = float(row["a2"])
            a3v = float(row["a3"])
            a1t, a2t, a3t = threshold_activities(a1v, a2v, a3v)
            c = (a1t * v1 + a2t * v2 + a3t * v3) / 3.0
            xs.append(float(c[0]))
            ys.append(float(c[1]))
        ax.scatter(xs, ys, s=36, color='black', alpha=0.9, zorder=2)

    # Draw axes last so they sit on top
    plot_axes(ax, max_len=axes_len)
    # Explicitly lock the view limits based on axes_len (avoid autoscaling changes)
    v1, v2, v3 = axis_vectors()
    origin = np.array([0.0, 0.0])
    endpoints = np.stack([
        origin + axes_len * v1,
        origin + axes_len * v2,
        origin + axes_len * v3,
        origin
    ], axis=0)
    margin = 0.05 * axes_len
    x_min, y_min = endpoints.min(axis=0) - margin
    x_max, y_max = endpoints.max(axis=0) + margin
    ax.set_autoscale_on(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Remove plot frame (black box)
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = "data_analysis/spec2_cumulative_centroid.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spec2 cumulative centroid plot to {out_path}")


if __name__ == "__main__":
    main()


