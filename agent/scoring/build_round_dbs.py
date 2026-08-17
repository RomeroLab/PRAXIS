#!/usr/bin/env python
"""
Build per-round truncated copies of the assayed-sequence database.

For the round-sweep analysis we want, for each round r, the labelled context the
round-r model was trained against. The DB has no round column, but sequences were
assayed in per-round batches recorded by ``created_at``: there are exactly 25
distinct ``created_at`` batches (1 seed + 24 agent rounds) with cumulative row
counts 6,12,18,...,132. These map 1:1 to the 25 deduplicated checkpoints
(``round_0``..``round_24``), so round r's context is simply the DB sliced at the
r-th batch boundary -- all rows with ``created_at`` at or before that batch's
timestamp. No per-round CSVs are read; everything comes from the DB's timestamps.

Each output is a full copy of the master DB (schema/indexes/table name intact)
with rows past the cutoff DELETEd, so it drops straight into the normal
``get_train_database`` path (``score_proteins.py --database <round_db>``).

    python scoring/build_round_dbs.py                # rounds 0..24 -> scoring/round_dbs/
    python scoring/build_round_dbs.py --rounds 1-24
"""
import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = str(_AGENT_DIR / "data" / "Bgl_spec_2026_final_database_final.db")
DEFAULT_OUT_DIR = str(_AGENT_DIR / "scoring" / "round_dbs")


def detect_table(con, db_path):
    """Return the single user table (mirrors score_proteins.detect_table)."""
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]
    if len(names) == 1:
        return names[0]
    sys.exit(f"Expected exactly one user table in {db_path}, found: {names}")


def parse_rounds(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master_db", default=DEFAULT_MASTER)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--rounds", default="0-24", help="e.g. '0-24' or '1,5,24'")
    args = p.parse_args()

    rounds = parse_rounds(args.rounds)
    os.makedirs(args.out_dir, exist_ok=True)

    src = sqlite3.connect(args.master_db)
    table = detect_table(src, args.master_db)
    # Ordered distinct assay batches (one per round that collected data).
    batches = [r[0] for r in src.execute(
        f"SELECT created_at FROM {table} GROUP BY created_at "
        f"ORDER BY created_at, MIN(sequence_id)"
    ).fetchall()]
    total = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    src.close()
    print(f"Master DB: {os.path.basename(args.master_db)} | table '{table}' | "
          f"{total} rows | {len(batches)} created_at batches")

    if len(rounds) > len(batches):
        sys.exit(f"Requested {len(rounds)} rounds but DB has only {len(batches)} "
                 f"created_at batches")
    for r in rounds:
        if r >= len(batches):
            sys.exit(f"Round {r} has no matching created_at batch "
                     f"(only {len(batches)}: rounds 0..{len(batches)-1})")
        cutoff = batches[r]
        out_path = os.path.join(args.out_dir, f"Bgl_spec_2026_round_{r}.db")
        shutil.copyfile(args.master_db, out_path)
        con = sqlite3.connect(out_path)
        con.execute(f"DELETE FROM {table} WHERE created_at > ?", (cutoff,))
        con.commit()
        kept = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute("VACUUM")
        con.close()
        print(f"  round {r:2d}: cutoff={cutoff}  rows={kept:3d}  "
              f"-> {os.path.basename(out_path)}")

    print(f"Done. {len(rounds)} round DBs in {args.out_dir}")


if __name__ == "__main__":
    main()
