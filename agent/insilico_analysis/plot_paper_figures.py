"""
Generate the fitness-trajectory figure from benchmark results.

Usage:
    python insilico_analysis/plot_paper_figures.py \
        --dataset_dir insilico_analysis/dataset_oracle \
        --output_dir insilico_analysis/figures

Reads per-seed round logs from {dataset_dir}/{protein}/results/*_seed{N}_{method}_round_log.csv
and the dataset pickle {dataset_dir}/{protein}/{protein}_SeqFxnDataset.pkl, and writes
fitness_trajectory.{pdf,png,svg} to output_dir.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def plot_fitness_trajectory(output_dir, proteins=None, dataset_dir=None):
    """Fitness trajectory: two rows per protein.
    Row 1: cumulative best fitness per round (median across seeds).
    Row 2: scatter of per-seed final best fitness, one vertical strip per method.
    """
    if proteins is None:
        proteins = PROTEINS
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"dataset_dir not found: {dataset_dir!r} (cwd: {os.getcwd()}). "
            "Run from the repo root, or pass an explicit --dataset_dir.")
    n = len(proteins)
    sq = 7
    fig, axes = plt.subplots(2, n, figsize=(sq * n, sq * 2),
                             sharex=False, sharey=False, squeeze=False)

    scatter_method_keys = ['full_pipeline', 'greedy', 'onehot_mlp', 'onehot_mlp_greedy', 'random']
    data_found = False  # set True once any method yields plottable data

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
            data_found = True
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

        # --- Row 2: median with jittered individual dots ---
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

    if not data_found:
        plt.close(fig)
        raise RuntimeError(
            f"No round-log data found under {dataset_dir!r} for proteins {proteins} "
            "(expected {protein}/results/*_seed{N}_{method}_round_log.csv). "
            "Refusing to write a blank figure.")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(output_dir, 'fitness_trajectory.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the fitness-trajectory figure from benchmark results')
    parser.add_argument('--output_dir', type=str, default='insilico_analysis/figures')
    parser.add_argument('--proteins', type=str, nargs='+', default=None,
                        help='Subset of proteins to plot (default: all four)')
    parser.add_argument('--dataset_dir', type=str, default='insilico_analysis/dataset_oracle',
                        help='Directory containing {protein}/results/ round logs and {protein}_SeqFxnDataset.pkl')
    args = parser.parse_args()

    proteins = [p.lower() for p in args.proteins] if args.proteins else PROTEINS
    os.makedirs(args.output_dir, exist_ok=True)
    plot_fitness_trajectory(args.output_dir, proteins=proteins, dataset_dir=args.dataset_dir)
    print("Figure generated.")
