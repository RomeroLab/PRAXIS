"""
Generate paper figures from aggregated benchmark results.

Usage:
    python insilico_analysis/plot_paper_figures.py \
        --results_base_dir insilico_analysis/aggregated \
        --output_dir insilico_analysis/figures

Expects directory structure:
    {results_base_dir}/{protein}/{method}/sample_efficiency.csv
    {results_base_dir}/{protein}/{method}/topk_discovery_rate.csv
    {results_base_dir}/{protein}/{method}/time_to_threshold.csv
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROTEINS = ['gfp', 'gb1', 'pab1', 'ube4']

# Colors matching the reference script styling
purple = '#5a63a4'
yellow = '#fda404'
red = '#fd4470'

METHODS = {
    'random':            {'label': 'Random', 'color': red, 'alpha': 1.0, 'linewidth': 5.0, 'ls': '-'},
    'onehot_mlp':        {'label': 'One-hot MLP + UCB', 'color': yellow, 'alpha': 1.0, 'linewidth': 5.0, 'ls': '-'},
    'onehot_mlp_greedy': {'label': 'One-hot MLP + Greedy', 'color': yellow, 'alpha': 0.35, 'linewidth': 5.0, 'ls': '-'},
    'greedy':            {'label': 'Greedy (ProteinNPT, no exploration)', 'color': purple, 'alpha': 0.35, 'linewidth': 5.0, 'ls': '-'},
    'full_pipeline':     {'label': 'Full Pipeline (ProteinNPT + UCB)', 'color': purple, 'alpha': 1.0, 'linewidth': 5.0, 'ls': '-'},
}


def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def style_ax(ax):
    """Apply reference-script formatting: thick spines, thick ticks, no grid."""
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(which='both', width=2, length=8)
    ax.grid(False)


def _load_dataset_range(protein, dataset_dir):
    """Load dataset PKL and return (min, max) of functional_score."""
    pkl_path = os.path.join(dataset_dir, protein, f'{protein}_SeqFxnDataset.pkl')
    if not os.path.exists(pkl_path):
        return None, None
    df = pd.read_pickle(pkl_path)
    scores = pd.to_numeric(df['functional_score'], errors='coerce').dropna()
    return float(scores.min()), float(scores.max())


def _load_round_best(protein, method, dataset_dir):
    """Load per-seed round logs and return a DataFrame with median best fitness per round."""
    results_dir = os.path.join(dataset_dir, f'{protein}', 'results')
    if not os.path.isdir(results_dir):
        return None
    seed_dfs = []
    for fname in os.listdir(results_dir):
        if re.search(rf'_seed\d+_{re.escape(method)}_round_log\.csv$', fname):
            df = pd.read_csv(os.path.join(results_dir, fname))
            best_per_round = df.groupby('round')['fitness'].max().reset_index()
            best_per_round = best_per_round.rename(columns={'fitness': 'best_fitness'})
            best_per_round['best_fitness'] = best_per_round['best_fitness'].cummax()
            seed_dfs.append(best_per_round)
    if not seed_dfs:
        return None
    # Forward-fill truncated seeds so every seed covers all rounds.
    # Without this, seeds that ended early silently drop out of later
    # rounds, changing the sample size for the median and causing rank
    # order inconsistencies with the per-seed scatter plot (row 2).
    max_round = max(df['round'].max() for df in seed_dfs)
    full_rounds = pd.DataFrame({'round': range(1, max_round + 1)})
    filled = []
    for df in seed_dfs:
        merged = full_rounds.merge(df, on='round', how='left')
        merged['best_fitness'] = merged['best_fitness'].ffill()
        filled.append(merged)
    combined = pd.concat(filled)
    agg = combined.groupby('round')['best_fitness'].agg(['median']).reset_index()
    agg.columns = ['round', 'mean_best_fitness']
    return agg


def _load_seed_finals(protein, method, dataset_dir):
    """Return list of final cumulative-best fitness values, one per seed."""
    results_dir = os.path.join(dataset_dir, f'{protein}', 'results')
    if not os.path.isdir(results_dir):
        return []
    finals = []
    for fname in sorted(os.listdir(results_dir)):
        if re.search(rf'_seed\d+_{re.escape(method)}_round_log\.csv$', fname):
            df = pd.read_csv(os.path.join(results_dir, fname))
            finals.append(float(df['fitness'].max()))
    return finals


def _load_round_mean(protein, method, dataset_dir):
    """Load per-seed round logs and return DataFrame with median-of-means fitness per round.

    For each seed: compute mean fitness per round (no cumulative).
    Then take the median across seeds at each round.
    """
    results_dir = os.path.join(dataset_dir, f'{protein}', 'results')
    if not os.path.isdir(results_dir):
        return None
    seed_dfs = []
    for fname in os.listdir(results_dir):
        if re.search(rf'_seed\d+_{re.escape(method)}_round_log\.csv$', fname):
            df = pd.read_csv(os.path.join(results_dir, fname))
            mean_per_round = df.groupby('round')['fitness'].mean().reset_index()
            mean_per_round = mean_per_round.rename(columns={'fitness': 'mean_fitness'})
            seed_dfs.append(mean_per_round)
    if not seed_dfs:
        return None
    max_round = max(df['round'].max() for df in seed_dfs)
    full_rounds = pd.DataFrame({'round': range(1, max_round + 1)})
    filled = []
    for df in seed_dfs:
        merged = full_rounds.merge(df, on='round', how='left')
        merged['mean_fitness'] = merged['mean_fitness'].ffill()
        filled.append(merged)
    combined = pd.concat(filled)
    agg = combined.groupby('round')['mean_fitness'].agg(['median']).reset_index()
    agg.columns = ['round', 'median_mean_fitness']
    return agg


def _load_seed_final_means(protein, method, dataset_dir):
    """Return list of mean fitness at the final round, one per seed."""
    results_dir = os.path.join(dataset_dir, f'{protein}', 'results')
    if not os.path.isdir(results_dir):
        return []
    finals = []
    for fname in sorted(os.listdir(results_dir)):
        if re.search(rf'_seed\d+_{re.escape(method)}_round_log\.csv$', fname):
            df = pd.read_csv(os.path.join(results_dir, fname))
            last_round = df['round'].max()
            finals.append(float(df[df['round'] == last_round]['fitness'].mean()))
    return finals


def plot_mean_fitness(results_base_dir, output_dir, proteins=None, dataset_dir=None):
    """Like fig1 but row 1 shows mean fitness per round (median across seeds)
    and row 2 shows per-seed final-round mean fitness."""
    if proteins is None:
        proteins = PROTEINS
    n = len(proteins)
    sq = 7
    fig, axes = plt.subplots(2, n, figsize=(sq * n, sq * 2),
                             sharex=False, sharey=False, squeeze=False)

    method_keys = list(METHODS.keys())
    scatter_method_keys = ['full_pipeline', 'greedy', 'onehot_mlp', 'onehot_mlp_greedy', 'random']

    for idx, protein in enumerate(proteins):
        ax_line = axes[0][idx]
        ax_scatter = axes[1][idx]
        ax_line.set_box_aspect(1)
        ax_scatter.set_box_aspect(3)

        # --- Row 1: line plot (mean fitness per round, median across seeds) ---
        for method, style in METHODS.items():
            df = _load_round_mean(protein, method, dataset_dir) if dataset_dir else None
            if df is None:
                continue
            rounds = df['round']
            y = df['median_mean_fitness']
            ax_line.plot(rounds, y,
                         style['ls'], color=style['color'], linewidth=style['linewidth'],
                         alpha=style['alpha'], label=style['label'])

        all_line_vals = []
        for method in METHODS:
            df = _load_round_mean(protein, method, dataset_dir) if dataset_dir else None
            if df is not None:
                all_line_vals.extend(df['median_mean_fitness'].dropna().tolist())
        if all_line_vals:
            y_bottom = min(all_line_vals)
            y_top = max(all_line_vals)
            y_range = y_top - y_bottom
            ax_line.set_ylim(y_bottom - 0.1 * y_range, y_top + 0.1 * y_range)

        max_round = 20
        for r in range(1, max_round + 1):
            ax_line.axvline(x=r, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5, zorder=0)
        ax_line.set_xticks(range(5, max_round + 1, 5))
        ax_line.set_xticklabels([str(x) for x in range(5, max_round + 1, 5)])
        ax_line.set_xticks(range(1, max_round + 1), minor=True)
        ax_line.set_xlabel('Round', fontsize=12)
        ax_line.set_ylabel('Mean Fitness', fontsize=12)
        ax_line.set_title(protein.upper(), fontsize=14, fontweight='bold')
        style_ax(ax_line)
        ax_line.tick_params(axis='x', which='major', length=8, width=2)
        ax_line.tick_params(axis='x', which='minor', length=6, width=1.5)

        def beeswarm_x(vals, x_center, radius=0.12):
            vals = np.array(vals, dtype=float)
            order = np.argsort(vals)
            xs = np.zeros(len(vals))
            placed = []
            offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4]
            for i in order:
                y = vals[i]
                for col in offsets:
                    x_off = col * radius
                    if not any(abs(y - py) < radius and abs(x_off - px) < radius
                               for py, px in placed):
                        xs[i] = x_center + x_off
                        placed.append((y, x_off))
                        break
                else:
                    xs[i] = x_center
                    placed.append((y, 0.0))
            return xs

        # --- Row 2: scatter of per-seed final-round mean fitness ---
        all_scatter_vals = []
        for m_idx, method in enumerate(scatter_method_keys):
            style = METHODS[method]
            finals = _load_seed_final_means(protein, method, dataset_dir) if dataset_dir else []
            if not finals:
                continue
            all_scatter_vals.extend(finals)
            x_pos = m_idx + 1

            xs = beeswarm_x(finals, x_pos)
            ax_scatter.scatter(xs, finals,
                               color=style['color'], alpha=style['alpha'], s=70, zorder=3,
                               label=style['label'])
            median_val = float(np.median(finals))
            ax_scatter.hlines(median_val, x_pos - 0.5, x_pos + 0.5,
                              color=style['color'], alpha=style['alpha'],
                              linewidth=4.0, zorder=2)

        if all_scatter_vals:
            y_bottom = min(all_scatter_vals)
            y_top = max(all_scatter_vals)
            y_range = y_top - y_bottom
            ax_scatter.set_ylim(y_bottom - 0.1 * y_range, y_top + 0.1 * y_range)

        for m_sep in range(1, len(scatter_method_keys)):
            ax_scatter.axvline(x=m_sep + 0.5, color='lightgray', linestyle='-',
                               alpha=0.5, linewidth=1.5, zorder=0)
        ax_scatter.set_xlim(0.5, len(scatter_method_keys) + 0.5)
        ax_scatter.set_xticks(range(1, len(scatter_method_keys) + 1))
        ax_scatter.set_xticklabels([METHODS[m]['label'] for m in scatter_method_keys],
                                   rotation=20, ha='right', fontsize=10)
        ax_scatter.set_ylabel('Mean Fitness', fontsize=12)
        ax_scatter.set_title(protein.upper(), fontsize=14, fontweight='bold')
        style_ax(ax_scatter)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(output_dir, 'fig1_mean_fitness.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sample_efficiency(results_base_dir, output_dir, proteins=None, dataset_dir=None):
    """Figure 1: two rows per protein.
    Row 1: cumulative best fitness per round (mean across seeds).
    Row 2: scatter of per-seed final best fitness, one vertical strip per method.
    """
    if proteins is None:
        proteins = PROTEINS
    n = len(proteins)
    sq = 7
    fig, axes = plt.subplots(2, n, figsize=(sq * n, sq * 2),
                             sharex=False, sharey=False, squeeze=False)

    method_keys = list(METHODS.keys())
    scatter_method_keys = ['full_pipeline', 'greedy', 'onehot_mlp', 'onehot_mlp_greedy', 'random']

    for idx, protein in enumerate(proteins):
        ax_line = axes[0][idx]
        ax_scatter = axes[1][idx]
        ax_line.set_box_aspect(1)
        ax_scatter.set_box_aspect(3)

        _, ds_max = (None, None)
        if dataset_dir:
            _, ds_max = _load_dataset_range(protein, dataset_dir)

        # --- Row 1: line plot ---
        for method, style in METHODS.items():
            df = _load_round_best(protein, method, dataset_dir) if dataset_dir else None
            if df is None:
                continue
            rounds = df['round']
            y = df['mean_best_fitness']
            ax_line.plot(rounds, y,
                         style['ls'], color=style['color'], linewidth=style['linewidth'],
                         alpha=style['alpha'], label=style['label'])

        all_line_vals = []
        for method in METHODS:
            df = _load_round_best(protein, method, dataset_dir) if dataset_dir else None
            if df is not None:
                all_line_vals.extend(df['mean_best_fitness'].dropna().tolist())
        if all_line_vals:
            y_bottom = min(all_line_vals)
            y_top = max(all_line_vals)
            y_range = y_top - y_bottom
            ax_line.set_ylim(y_bottom - 0.1 * y_range, y_top + 0.1 * y_range)

        max_round = 20
        for r in range(1, max_round + 1):
            ax_line.axvline(x=r, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5, zorder=0)
        ax_line.set_xticks(range(5, max_round + 1, 5))
        ax_line.set_xticklabels([str(x) for x in range(5, max_round + 1, 5)])
        ax_line.set_xticks(range(1, max_round + 1), minor=True)
        ax_line.set_xlabel('Round', fontsize=12)
        ax_line.set_ylabel('Best Fitness', fontsize=12)
        ax_line.set_title(protein.upper(), fontsize=14, fontweight='bold')
        style_ax(ax_line)
        ax_line.tick_params(axis='x', which='major', length=8, width=2)
        ax_line.tick_params(axis='x', which='minor', length=6, width=1.5)

        def beeswarm_x(vals, x_center, radius=0.12):
            """Stack points horizontally when they overlap vertically."""
            vals = np.array(vals, dtype=float)
            order = np.argsort(vals)
            xs = np.zeros(len(vals))
            placed = []  # list of (y, x_offset) already placed
            offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4]  # alternating sides
            for i in order:
                y = vals[i]
                for col in offsets:
                    x_off = col * radius
                    if not any(abs(y - py) < radius and abs(x_off - px) < radius
                               for py, px in placed):
                        xs[i] = x_center + x_off
                        placed.append((y, x_off))
                        break
                else:
                    xs[i] = x_center
                    placed.append((y, 0.0))
            return xs

        # --- Row 2: mean ± std with jittered individual dots ---
        all_scatter_vals = []
        for m_idx, method in enumerate(scatter_method_keys):
            style = METHODS[method]
            finals = _load_seed_finals(protein, method, dataset_dir) if dataset_dir else []
            if not finals:
                continue
            all_scatter_vals.extend(finals)
            x_pos = m_idx + 1

            xs = beeswarm_x(finals, x_pos)
            ax_scatter.scatter(xs, finals,
                               color=style['color'], alpha=style['alpha'], s=70, zorder=3,
                               label=style['label'])
            median_val = float(np.median(finals))
            ax_scatter.hlines(median_val, x_pos - 0.5, x_pos + 0.5,
                              color=style['color'], alpha=style['alpha'],
                              linewidth=4.0, zorder=2)

        if all_scatter_vals:
            y_bottom = min(all_scatter_vals)
            y_top = max(all_scatter_vals)
            y_range = y_top - y_bottom
            ax_scatter.set_ylim(y_bottom - 0.1 * y_range, y_top + 0.1 * y_range)

        for m_sep in range(1, len(scatter_method_keys)):
            ax_scatter.axvline(x=m_sep + 0.5, color='lightgray', linestyle='-',
                               alpha=0.5, linewidth=1.5, zorder=0)
        ax_scatter.set_xlim(0.5, len(scatter_method_keys) + 0.5)
        ax_scatter.set_xticks(range(1, len(scatter_method_keys) + 1))
        ax_scatter.set_xticklabels([METHODS[m]['label'] for m in scatter_method_keys],
                                   rotation=20, ha='right', fontsize=10)
        ax_scatter.set_ylabel('Best Fitness', fontsize=12)
        ax_scatter.set_title(protein.upper(), fontsize=14, fontweight='bold')
        style_ax(ax_scatter)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(output_dir, 'fig1_sample_efficiency.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_topk_discovery(results_base_dir, output_dir, k_values=[10, 100], proteins=None):
    """Figure 2: Top-K discovery rate. One panel per protein, lines per method."""
    if proteins is None:
        proteins = PROTEINS
    n_k = len(k_values)
    fig, axes = plt.subplots(len(proteins), n_k, figsize=(10 * n_k, 6 * len(proteins)),
                             sharex=False, sharey=True, squeeze=False)

    for p_idx, protein in enumerate(proteins):
        for k_idx, k in enumerate(k_values):
            ax = axes[p_idx][k_idx]
            for method, style in METHODS.items():
                path = os.path.join(results_base_dir, protein, method, 'topk_discovery_rate.csv')
                df = load_csv_safe(path)
                if df is None:
                    continue
                dk = df[df['k'] == k]
                if dk.empty:
                    continue
                ax.plot(dk['round'], dk['mean_fraction'],
                        style['ls'], color=style['color'], linewidth=style['linewidth'],
                        alpha=style['alpha'], label=style['label'])

            # Vertical round lines
            rounds = list(range(1, 21))
            for r in rounds:
                ax.axvline(x=r, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5)
            even_xticks = [x for x in rounds if x % 2 == 0]
            ax.set_xticks(even_xticks)
            ax.set_xticklabels([str(x) for x in even_xticks])

            ax.set_title(f"{protein.upper()} - Top-{k}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Fraction Discovered', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            style_ax(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(output_dir, 'fig2_topk_discovery.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_time_to_threshold(results_base_dir, output_dir, thresholds=[0.80, 0.90, 0.95], proteins=None):
    """Figure 3: Grouped bar chart. X: threshold, Y: queries needed."""
    if proteins is None:
        proteins = PROTEINS
    fig, axes = plt.subplots(1, len(proteins), figsize=(8 * len(proteins), 6), sharey=False)
    if len(proteins) == 1:
        axes = [axes]

    bar_width = 0.18
    method_list = list(METHODS.keys())

    for p_idx, protein in enumerate(proteins):
        ax = axes[p_idx]
        x = np.arange(len(thresholds))
        for m_idx, method in enumerate(method_list):
            path = os.path.join(results_base_dir, protein, method, 'time_to_threshold.csv')
            df = load_csv_safe(path)
            if df is None:
                means = [float('nan')] * len(thresholds)
                stds = [0.0] * len(thresholds)
            else:
                means = []
                stds = []
                for t in thresholds:
                    row = df[df['threshold'].astype(float) == t]
                    if row.empty:
                        means.append(float('nan'))
                        stds.append(0.0)
                    else:
                        means.append(row['mean_queries'].values[0])
                        stds.append(row['std'].values[0] if 'std' in row.columns else 0.0)
            offset = (m_idx - len(method_list) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                          label=METHODS[method]['label'], color=METHODS[method]['color'],
                          alpha=METHODS[method]['alpha'], capsize=3)

        ax.set_title(protein.upper(), fontsize=14, fontweight='bold')
        ax.set_xlabel('Threshold (fraction of global max)', fontsize=12)
        ax.set_ylabel('Queries Needed', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0%}" for t in thresholds])
        style_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(output_dir, 'fig3_time_to_threshold.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate paper figures from aggregated benchmark results')
    parser.add_argument('--results_base_dir', type=str, default='insilico_analysis/aggregated',
                        help='Base directory containing {protein}/{method}/ subdirs with aggregated CSVs')
    parser.add_argument('--output_dir', type=str, default='insilico_analysis/figures')
    parser.add_argument('--proteins', type=str, nargs='+', default=None,
                        help='Subset of proteins to plot (default: all four)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 100])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.80, 0.90, 0.95])
    parser.add_argument('--dataset_dir', type=str, default='insilico_analysis/dataset_oracle',
                        help='Directory containing {protein}/{protein}_SeqFxnDataset.pkl for y-axis normalization')
    args = parser.parse_args()

    if args.proteins:
        proteins = [p.lower() for p in args.proteins]
    else:
        proteins = PROTEINS

    os.makedirs(args.output_dir, exist_ok=True)

    plot_sample_efficiency(args.results_base_dir, args.output_dir, proteins=proteins, dataset_dir=args.dataset_dir)
    plot_mean_fitness(args.results_base_dir, args.output_dir, proteins=proteins, dataset_dir=args.dataset_dir)
    plot_topk_discovery(args.results_base_dir, args.output_dir, k_values=args.k_values, proteins=proteins)
    plot_time_to_threshold(args.results_base_dir, args.output_dir, thresholds=args.thresholds, proteins=proteins)
    print("All figures generated.")
