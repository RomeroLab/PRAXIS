# In-silico analysis utilities

This folder contains protein-agnostic utilities to inspect datasets, rank top sequences, and analyze active-learning progress across rounds.

## Shared utilities

- `insilico_analysis/common.py`: helper functions for sequence normalization, mutation parsing, accessibility checks, and loading results from CSV/PKL/SQLite.

## Scripts

- `insilico_analysis/gb1/top10_from_results.py`
  - Input: any CSV/PKL/SQLite table with columns including `mutations`, a sequence column (e.g., `sequence`), and a phenotype column (default `fitness`).
  - Flags: `--zscore`, `--wt`, `--wt_score`, `--accessible_only`, `--include_stops`, `--min_value`, `--rank_seq`.
  - Example:
    ```bash
    python3 insilico_analysis/gb1/top10_from_results.py \
      --pkl /path/to/SeqFxnDataset.pkl \
      --phenotype functional_score \
      --zscore --accessible_only --min_value 0 \
      --wt "<WT_SEQUENCE>" --wt_score <WT_VALUE>
    ```

- `insilico_analysis/gb1/plot_avg_fitness_over_rounds.py`
  - Plots per-round best accessible functional score for cumulative and newly added sequences. Supports per-round CSVs named `*_round_<N>_training_data.csv`.
  - Flags: `--rounds_dir` (preferred) or `--csv` with `--first_round_size`/`--round_size`, `--round_number_offset`, `--num_rounds`.
  - Example:
    ```bash
    python3 insilico_analysis/gb1/plot_avg_fitness_over_rounds.py \
      --rounds_dir /path/to/model_checkpoints/<protein>_frontier \
      --pkl /path/to/SeqFxnDataset.pkl \
      --num_rounds 20 --round_number_offset 1
    ```

## Porting to other proteins

- Provide a PKL with at least: `sequence`, `mutations`, and the phenotype column (e.g., `functional_score`).
- If using round-wise analysis, ensure per-round CSVs contain a `sequence` column and follow the `*_round_<N>_training_data.csv` naming.
- Update phenotype flag (e.g., `--phenotype stability_score`) as needed. 