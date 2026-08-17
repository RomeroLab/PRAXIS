import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, percentileofscore
from pathlib import Path
import re
import itertools

# Removed dependency on insilico_analysis.common accessible utilities
# Provide a simple local normalizer

def normalize_sequence(s: str) -> str:
	return str(s).replace('*', '')


def compute_round_index(row_index: int, first_round_size: int, round_size: int) -> int:
	if row_index < first_round_size:
		return 1
	rem = row_index - first_round_size
	return 2 + (rem // round_size)


def main():
	parser = argparse.ArgumentParser(description="Plot best (max) functional score per round (accessible sequences only)")
	parser.add_argument("--csv", help="Path to a training data CSV (used only if --rounds_dir not provided; first 11 rows round 1, next 10 rows round 2, etc.)")
	parser.add_argument("--rounds_dir", help="Directory containing per-round CSVs named with pattern '*_round_<N>_training_data.csv'")
	parser.add_argument("--pkl", required=True, help="Path to dataset PKL containing 'sequence', 'functional_score', and 'mutations'")
	parser.add_argument("--first_round_size", type=int, default=10, help="If using --csv only: number of rows in the first round (default: 11)")
	parser.add_argument("--round_size", type=int, default=10, help="If using --csv only: rows per subsequent round (default: 10)")
	parser.add_argument("--num_rounds", type=int, help="Total number of rounds to display (optional; if omitted, inferred from data)")
	parser.add_argument("--round_number_offset", type=int, default=1, help="Offset added to round numbers parsed from filenames (default: 1 for 0-indexed filenames)")
	parser.add_argument("--out", help="Output PNG path (default: alongside the CSV/dir)")
	args = parser.parse_args()

	# Load PKL
	pkl_path = Path(args.pkl)
	pkl_df = pd.read_pickle(pkl_path)
	required_cols = {'sequence', 'functional_score'}
	if not required_cols.issubset(set(pkl_df.columns)):
		raise RuntimeError("PKL must contain 'sequence' and 'functional_score' columns")
	pkl_df_norm = pkl_df[['sequence', 'functional_score']].copy()
	pkl_df_norm['sequence_norm'] = pkl_df_norm['sequence'].astype(str).map(normalize_sequence)
	# Dataset-wide max functional score (unfiltered)
	pkl_max = pd.to_numeric(pkl_df_norm['functional_score'], errors='coerce').max()
	print(f"Dataset max functional_score (PKL): {pkl_max}")
	# Dataset size in number of sequences (rows) and unique normalized sequences
	num_sequences = len(pkl_df_norm)
	num_unique_sequences = pkl_df_norm['sequence_norm'].nunique()
	print(f"Dataset PKL size: {num_sequences} sequences ({num_unique_sequences} unique by normalized sequence)")
	# Prepare scores for histogram (drop NaNs)
	scores = pd.to_numeric(pkl_df_norm['functional_score'], errors='coerce').dropna()

	round_stats = None

	# Preferred mode: scan rounds_dir for per-round CSVs
	if args.rounds_dir:
		rounds_dir = Path(args.rounds_dir)
		if not rounds_dir.is_dir():
			raise RuntimeError(f"--rounds_dir not found: {rounds_dir}")
		rows = []
		pattern = re.compile(r"_round_(\d+)_training_data\.csv$")
		seen = set()
		# Collect (parsed_round, round_num, path) and sort numerically by parsed_round
		files = []
		for p in rounds_dir.glob("*.csv"):
			m = pattern.search(p.name)
			if m:
				parsed_round = int(m.group(1))
				round_num = parsed_round + args.round_number_offset
				files.append((parsed_round, round_num, p))
		files.sort(key=lambda x: x[0])
		for parsed_round, round_num, csv_file in files:
			csv_df = pd.read_csv(csv_file)
			if 'sequence' not in csv_df.columns:
				print(f"Warning: skipping {csv_file.name} (no 'sequence' column)")
				continue
			csv_df = csv_df.copy()
			csv_df['sequence_norm'] = csv_df['sequence'].astype(str).map(normalize_sequence)
			merged = csv_df.merge(pkl_df_norm[['sequence_norm', 'functional_score']], on='sequence_norm', how='left')
			# Best cumulative among sequences present in this (cumulative) round file (no accessibility filter)
			best_cum = pd.to_numeric(merged['functional_score'], errors='coerce').max()
			# Best and average among newly appeared sequences in this round (not seen in previous rounds)
			curr_set = set(merged['sequence_norm'].astype(str))
			new_set = curr_set - seen
			if new_set:
				merged_new = merged.loc[merged['sequence_norm'].isin(new_set)]
				best_new = pd.to_numeric(merged_new['functional_score'], errors='coerce').max()
				avg_new = pd.to_numeric(merged_new['functional_score'], errors='coerce').mean()
			else:
				best_new = float('nan')
				avg_new = float('nan')
			rows.append((round_num, best_cum, best_new, avg_new))
			seen = curr_set
		round_stats = pd.DataFrame(rows, columns=['round', 'best_cumulative', 'best_new', 'avg_new']).sort_values('round').reset_index(drop=True)
		default_out_base = rounds_dir
	else:
		# Fallback mode: single CSV with row chunking
		if not args.csv:
			raise RuntimeError("Provide either --rounds_dir or --csv")
		csv_path = Path(args.csv)
		csv_df = pd.read_csv(csv_path)
		if 'sequence' not in csv_df.columns:
			raise RuntimeError("CSV missing 'sequence' column")
		csv_df = csv_df.copy()
		csv_df['sequence_norm'] = csv_df['sequence'].astype(str).map(normalize_sequence)
		merged = csv_df.merge(pkl_df_norm[['sequence_norm', 'functional_score']], on='sequence_norm', how='left')
		merged = merged.reset_index(drop=True)
		merged['round'] = [compute_round_index(i, args.first_round_size, args.round_size) for i in range(len(merged))]
		rows = []
		seen = set()
		for r in sorted(merged['round'].unique()):
			curr = merged[merged['round'] == r]
			best_cum = pd.to_numeric(merged.loc[merged['round'] <= r, 'functional_score'], errors='coerce').max()
			curr_set = set(curr['sequence_norm'].astype(str))
			new_set = curr_set - seen
			if new_set:
				best_new = pd.to_numeric(curr.loc[curr['sequence_norm'].isin(new_set), 'functional_score'], errors='coerce').max()
				avg_new = pd.to_numeric(curr.loc[curr['sequence_norm'].isin(new_set), 'functional_score'], errors='coerce').mean()
			else:
				best_new = float('nan')
				avg_new = float('nan')
			rows.append((r, best_cum, best_new, avg_new))
			seen |= curr_set
		round_stats = pd.DataFrame(rows, columns=['round', 'best_cumulative', 'best_new', 'avg_new']).sort_values('round').reset_index(drop=True)
		default_out_base = csv_path.parent

	# Reindex rounds
	if args.num_rounds:
		all_rounds = pd.Index(range(1, args.num_rounds + 1), name='round')
		round_stats = round_stats.set_index('round').reindex(all_rounds)
		for col in ['best_cumulative', 'best_new', 'avg_new']:
			if round_stats[col].isna().any():
				missing_rounds = round_stats.index[round_stats[col].isna()].tolist()
				print(f"Note: No data for {col} in rounds: {missing_rounds}")
		round_stats = round_stats.reset_index()

	print("Per-round functional_score (cumulative best, best new, average new):")
	print(round_stats.to_string(index=False))

	# Percentile of the best cumulative score within all scores in the PKL
	try:
		best_cum_overall = pd.to_numeric(round_stats['best_cumulative'], errors='coerce').max()
		best_cum_percentile = percentileofscore(scores, best_cum_overall, kind='weak')
		print(f"Best cumulative score: {best_cum_overall} (percentile among PKL scores: {best_cum_percentile:.2f}th)")
	except Exception as e:
		print(f"Warning: could not compute percentile for best cumulative score: {e}")

	# Plot
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [4, 0.7], 'wspace': 0.2})
	blue = '#5a63a4'
	yellow = '#fda404'
	# Cumulative best: solid blue, thicker, higher alpha (left subplot)
	ax1.plot(round_stats['round'], round_stats['best_cumulative'], '-', color=blue, linewidth=5.0, alpha=0.9, label='Cumulative best')
	# Best new this round: semi-transparent blue underlay
	ax1.plot(round_stats['round'], round_stats['best_new'], '-', color=blue, linewidth=3.0, alpha=0.25, label='Best new this round')
	# Average new this round: yellow
	ax1.plot(round_stats['round'], round_stats['avg_new'], '-', color=yellow, linewidth=3.0, alpha=0.9, label='Avg new this round')
	ax1.set_xlabel('Round')
	ax1.set_ylabel('functional_score')
	ax1.set_title('Fitness by round')
	# Force integer round ticks and style axes on left subplot
	if args.num_rounds:
		xticks = list(range(1, args.num_rounds + 1))
	else:
		valid_rounds = sorted(set(int(r) for r in round_stats['round'].dropna().tolist()))
		xticks = valid_rounds
	# Show only even-numbered rounds: 2, 4, 6, ...
	even_xticks = [x for x in xticks if x % 2 == 0]
	ax1.set_xticks(even_xticks)
	ax1.set_xticklabels([str(x) for x in even_xticks])
	# Thicker borders and ticks
	for spine in ax1.spines.values():
		spine.set_linewidth(3)
	ax1.tick_params(which='both', width=2, length=8)
	# Vertical lines at each round
	for r in xticks:
		ax1.axvline(x=r, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5)
	# Red dotted reference line at functional_score = 0.0
	red = '#fd4470'
	ax1.axhline(y=0.0, color=red, linestyle='--', linewidth=3, alpha=0.7)
	# Green dotted line at dataset-wide max functional_score from PKL
	#green = '#2ca02c'
	#ax1.axhline(y=pkl_max, color=green, linestyle='--', linewidth=3, alpha=0.7)

	# Right subplot: histogram of all functional_score values, oriented horizontally
	# Smooth density (KDE) instead of histogram
	kde = gaussian_kde(scores)
	# Use a dense, evenly spaced grid with slight padding on the min/max
	y_min, y_max = float(scores.min()), float(scores.max())
	pad = 0.02 * (y_max - y_min)
	y_grid = np.linspace(y_min - pad, y_max + pad, 2000)
	density = kde(y_grid)
	ax2.plot(density, y_grid, color='#a9a9a9', linewidth=2.5)
	ax2.set_ylim(y_min - pad, y_max + pad)
	ax2.set_xlabel('')
	ax2.set_xticks([])
	# Spines: keep only left
	for side in ['top', 'right', 'bottom']:
		ax2.spines[side].set_visible(False)
	ax2.spines['left'].set_linewidth(3)
	ax2.tick_params(axis='y', which='both', width=2, length=8)


	# No gridlines
	plt.tight_layout()

	# Output path
	if args.out:
		out_path = Path(args.out)
	else:
		out_path = default_out_base / 'best_functional_score_by_round.png'
	plt.savefig(out_path, dpi=150)
	print(f"Saved plot to: {out_path}")


if __name__ == '__main__':
	main() 