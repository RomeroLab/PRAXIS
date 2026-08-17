"""
Aggregate multi-seed benchmark results and compute metrics with confidence intervals.

Usage:
    python insilico_analysis/aggregate_seeds.py \
        --round_log_pattern "data/insilico/gfp/results/*_round_log.csv" \
        --dataset_csv /path/to/gfp_dataset.csv \
        --phenotype_col fitness \
        --output_dir insilico_analysis/aggregated/gfp
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from metrics import compute_topk_discovery_rate, compute_time_to_threshold, compute_sample_efficiency


def aggregate_from_round_logs(
    round_log_paths: list,
    dataset_df: pd.DataFrame,
    phenotype_col: str,
    seq_col: str = "sequence",
    k_values: list = [10, 50, 100],
    thresholds: list = [0.80, 0.90, 0.95, 0.99],
    output_dir: str = "aggregated",
):
    os.makedirs(output_dir, exist_ok=True)

    all_topk = []
    all_ttt = []
    all_efficiency = []

    for path in round_log_paths:
        rl = pd.read_csv(path)
        if phenotype_col not in rl.columns:
            # Try to find the phenotype column
            candidates = [c for c in rl.columns if c not in ('round', 'sequence')]
            if candidates:
                phenotype_col_local = candidates[0]
            else:
                print(f"Warning: skipping {path}, no phenotype column found")
                continue
        else:
            phenotype_col_local = phenotype_col

        topk = compute_topk_discovery_rate(rl, dataset_df, phenotype_col_local, seq_col=seq_col, k_values=k_values)
        topk['seed_file'] = os.path.basename(path)
        all_topk.append(topk)

        ttt = compute_time_to_threshold(rl, dataset_df, phenotype_col_local, seq_col=seq_col, thresholds=thresholds)
        ttt_row = {'seed_file': os.path.basename(path)}
        ttt_row.update(ttt)
        all_ttt.append(ttt_row)

        eff = compute_sample_efficiency(rl, phenotype_col_local)
        eff['seed_file'] = os.path.basename(path)
        all_efficiency.append(eff)

    if not all_topk:
        print("No valid round logs found.")
        return

    # Aggregate top-K discovery rate
    topk_all = pd.concat(all_topk, ignore_index=True)
    topk_agg = topk_all.groupby(['round', 'k'])['fraction_discovered'].agg(['mean', 'std', 'count']).reset_index()
    topk_agg['ci_half'] = 1.96 * topk_agg['std'] / np.sqrt(topk_agg['count'].clip(lower=1))
    topk_agg['ci_lower'] = topk_agg['mean'] - topk_agg['ci_half']
    topk_agg['ci_upper'] = topk_agg['mean'] + topk_agg['ci_half']
    topk_agg.rename(columns={'mean': 'mean_fraction'}, inplace=True)
    topk_agg.to_csv(os.path.join(output_dir, 'topk_discovery_rate.csv'), index=False)

    # Aggregate time-to-threshold
    ttt_df = pd.DataFrame(all_ttt)
    ttt_cols = [c for c in ttt_df.columns if c != 'seed_file']
    ttt_summary = []
    for col in ttt_cols:
        vals = ttt_df[col].dropna()
        ttt_summary.append({
            'threshold': col,
            'mean_queries': vals.mean() if len(vals) > 0 else float('nan'),
            'std': vals.std() if len(vals) > 1 else 0.0,
            'n_seeds_reached': len(vals),
        })
    pd.DataFrame(ttt_summary).to_csv(os.path.join(output_dir, 'time_to_threshold.csv'), index=False)

    # Aggregate sample efficiency
    eff_all = pd.concat(all_efficiency, ignore_index=True)
    eff_agg = eff_all.groupby('num_queries')['best_fitness'].agg(['mean', 'std', 'count']).reset_index()
    eff_agg.rename(columns={'mean': 'mean_best_fitness'}, inplace=True)
    eff_agg.to_csv(os.path.join(output_dir, 'sample_efficiency.csv'), index=False)

    print(f"Aggregated results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate multi-seed benchmark results')
    parser.add_argument('--round_log_pattern', type=str, required=True, help='Glob pattern for round log CSVs')
    parser.add_argument('--dataset_csv', type=str, required=True, help='Path to full dataset CSV')
    parser.add_argument('--phenotype_col', type=str, default='fitness', help='Phenotype column name in round logs and dataset')
    parser.add_argument('--dataset_phenotype_col', type=str, default=None,
                        help='Phenotype column name in the dataset if different from --phenotype_col (e.g. functional_score)')
    parser.add_argument('--seq_col', type=str, default='sequence', help='Sequence column name in dataset')
    parser.add_argument('--output_dir', type=str, default='insilico_analysis/aggregated', help='Output directory')
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 50, 100])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.80, 0.90, 0.95, 0.99])
    args = parser.parse_args()

    paths = sorted(glob.glob(args.round_log_pattern))
    if not paths:
        print(f"No files matched pattern: {args.round_log_pattern}")
        sys.exit(1)
    print(f"Found {len(paths)} round log files")

    if args.dataset_csv.endswith('.pkl') or args.dataset_csv.endswith('.pickle'):
        dataset_df = pd.read_pickle(args.dataset_csv)
    else:
        dataset_df = pd.read_csv(args.dataset_csv)

    # Rename dataset phenotype column if different from round-log phenotype column
    if args.dataset_phenotype_col and args.dataset_phenotype_col != args.phenotype_col:
        if args.dataset_phenotype_col in dataset_df.columns:
            dataset_df = dataset_df.rename(columns={args.dataset_phenotype_col: args.phenotype_col})
            print(f"Renamed dataset column '{args.dataset_phenotype_col}' -> '{args.phenotype_col}'")

    aggregate_from_round_logs(
        round_log_paths=paths,
        dataset_df=dataset_df,
        phenotype_col=args.phenotype_col,
        seq_col=args.seq_col,
        k_values=args.k_values,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
    )
