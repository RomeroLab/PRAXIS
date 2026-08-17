#!/usr/bin/env python
"""
Generate random chimeras by stitching sequence segments.

Reads the fragment table (``agent/data/sequence_segments.csv``: rows = parents,
columns = fragment positions f0..fN, each cell an amino-acid subsequence) and
builds random chimeras by choosing one parent's fragment at each position and
concatenating them in order.

Each chimera is identified by its block pattern: the parent number chosen at each
fragment position, left to right. With 6 parents and 8 fragments, pattern
``31245612`` means parent 3 at f0, parent 1 at f1, ... parent 2 at f7.

Writes a CSV with columns ``chimera`` (the block pattern) and ``sequence`` (the
stitched amino-acid sequence, trailing stop ``*`` stripped), ready to feed to
score_proteins.py. Sampling is reproducible via --seed; sequences are unique and,
by default, exclude anything already present in the assayed-sequence context DB
(so predictions are for genuinely unseen chimeras).

Light dependencies only (pandas / numpy / sqlite3) -- no torch -- so it runs on a
login node.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AGENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTS = str(_AGENT_DIR / "data" / "sequence_segments.csv")
DEFAULT_DATABASE = str(_AGENT_DIR / "data" / "Bgl_spec_2026_final_database_final.db")


def load_segments(path):
    """Return (parents, fragment_cols, grid) where grid[frag][parent_idx] = subseq."""
    df = pd.read_csv(path, index_col=0)
    parents = list(df.index)
    frag_cols = list(df.columns)
    # Full grid expected (every parent has every fragment); guard anyway.
    if df.isna().any().any():
        sys.exit(f"{path} has missing fragment cells; expected a full parent x fragment grid.")
    grid = df.to_numpy(dtype=object)  # shape (n_parents, n_frag)
    return parents, frag_cols, grid


def existing_sequences(db_path, table_name=None):
    """Return a set of clean (stop-stripped) sequences already in the context DB."""
    if not db_path or not os.path.exists(db_path):
        return set()
    con = sqlite3.connect(db_path)
    try:
        if table_name is None:
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'").fetchall()
            names = [r[0] for r in rows]
            if len(names) != 1:
                # Don't guess across multiple tables; just skip exclusion.
                print(f"Note: {len(names)} tables in DB; skipping DB-exclusion "
                      f"(pass --exclude_table to enable).")
                return set()
            table_name = names[0]
        seqs = con.execute(f"SELECT DISTINCT sequence FROM {table_name}").fetchall()
    finally:
        con.close()
    return {str(s[0]).replace("*", "").strip() for s in seqs if s[0] is not None}


def main():
    parser = argparse.ArgumentParser(
        description="Generate random chimeras from sequence segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS,
                        help="Fragment table CSV (parents x fragment positions).")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of unique chimeras to generate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible sampling.")
    parser.add_argument("--output", required=True,
                        help="Output CSV (columns: chimera, sequence).")
    parser.add_argument("--exclude_db", default=DEFAULT_DATABASE,
                        help="Context DB whose assayed sequences are excluded from "
                             "the sample. Pass '' to disable exclusion.")
    parser.add_argument("--exclude_table", default=None,
                        help="Table in --exclude_db (auto-detected if single table).")
    args = parser.parse_args()

    parents, frag_cols, grid = load_segments(args.segments)
    n_parents, n_frag = grid.shape
    total = n_parents ** n_frag
    print(f"Segments: {n_parents} parents x {n_frag} fragments "
          f"({parents}) -> {total:,} possible chimeras")
    if args.n > total:
        sys.exit(f"Requested --n {args.n} exceeds the {total:,} possible chimeras.")

    exclude = existing_sequences(args.exclude_db, args.exclude_table)
    if exclude:
        print(f"Excluding {len(exclude)} sequences already in the context DB.")

    rng = np.random.default_rng(args.seed)
    seen_patterns = set()
    seen_sequences = set()
    chimeras, sequences = [], []

    # Oversample in batches to absorb duplicate/excluded rejects; loop until we
    # have --n unique, novel chimeras (or exhaust a generous attempt budget).
    attempts = 0
    max_attempts = max(50 * args.n, 100_000)
    while len(chimeras) < args.n and attempts < max_attempts:
        batch = rng.integers(0, n_parents, size=(max(args.n, 1024), n_frag))
        for choice in batch:
            attempts += 1
            pattern = "".join(str(p + 1) for p in choice)  # parents numbered 1..n
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)
            seq = "".join(grid[choice[f], f] for f in range(n_frag)).replace("*", "").strip()
            if seq in seen_sequences or seq in exclude:
                continue
            seen_sequences.add(seq)
            chimeras.append(pattern)
            sequences.append(seq)
            if len(chimeras) >= args.n:
                break

    if len(chimeras) < args.n:
        sys.exit(f"Only produced {len(chimeras)}/{args.n} unique novel chimeras "
                 f"after {attempts:,} attempts; loosen exclusion or lower --n.")

    out = pd.DataFrame({"chimera": chimeras, "sequence": sequences})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    lengths = out["sequence"].str.len()
    print(f"Wrote {len(out)} chimeras -> {args.output}")
    print(f"Sequence length: min={lengths.min()} max={lengths.max()} "
          f"(chimeras vary in length when parents differ in fragment length)")


if __name__ == "__main__":
    main()
