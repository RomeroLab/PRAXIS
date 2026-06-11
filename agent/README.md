# PRAXIS — Agent (model + Bayesian optimization)

The **agent** half of PRAXIS: a self-driving protein-engineering agent that uses
Bayesian optimization over a [ProteinNPT](https://github.com/OATML-Markslab/ProteinNPT)
surrogate to iteratively propose enzyme variants with improved activity/specificity.
It closes the design–test loop with the [`environment/`](../environment) wet-lab automation system.

The loop:

1. Train a ProteinNPT model on sequences with measured phenotypes (stored in SQLite).
2. Generate candidate sequences via conditional sampling (masking/unmasking segments).
3. Score candidates with the model + UCB acquisition (MC-dropout uncertainty).
4. Send the top candidates to the lab (`environment/`) over HTTP/SSH.
5. Monitor for returned phenotype data, write results to the DB, and repeat.

## Setup

```bash
# 1. Create the conda environment (installs proteinnpt==1.5.1 and all deps via pip)
conda env create -f self_driving_env.yml
conda activate self_driving_env

# 2. Point PRAXIS_ROOT at this directory (all scripts auto-resolve it, but you can set it explicitly)
export PRAXIS_ROOT="$(pwd)"

# 3. Download model weights (ESM2-650M + Tranception Large) into ESM/ and Tranception/
bash setup.sh

# 4. Precompute the candidate space, zero-shot fitness, and embeddings
bash get_all_segment_combinations.sh
bash precompute_zero_shot.sh
bash precompute_embeddings.sh           # or precompute_embeddings_distributed.sh
```

Large model weights, embeddings, and result databases are **not** committed — see the
[data availability](../README.md#data--weights) section in the top-level README.

## Running the closed-loop agent

Configure `config.sh` first (`run_mode`, `run_id`, BO hyperparameters). `PRAXIS_ROOT`
auto-resolves to this directory; everything else is derived from it.

```bash
bash self_driving_single.sh      # one acquisition strategy
bash self_driving_multiple.sh    # three specificity agents (spec1/spec2/spec3) in parallel
```

Replay training offline from an existing database:

```bash
bash train_from_database_multiple.sh
```

## In-silico benchmarks

A single benchmark per protein, validating the agent against published deep-mutational-scanning
(DMS) landscapes used as ground-truth oracles. Each benchmark **initializes with 10 random
sequences from the dataset** and runs **5 seeds × 5 search methods**
(`full_pipeline`, `greedy`, `onehot_mlp`, `onehot_mlp_greedy`, `random`) over four proteins
(GFP, GB1, PAB1, UBE4).

```bash
# All proteins × methods × seeds on GPU 0 (completed runs auto-skip)
bash insilico_analysis/run_full_benchmark.sh 0

# One protein / one method
bash insilico_analysis/run_full_benchmark.sh 0 gfp
bash insilico_analysis/run_full_benchmark.sh 1 gb1 greedy

# Aggregate + plot
python insilico_analysis/benchmark_analysis.py
python insilico_analysis/plot_paper_figures.py
```

The DMS oracle datasets (`insilico_analysis/dataset_oracle/<protein>/<protein>_SeqFxnDataset.pkl`)
are bundled in the repository, so the benchmarks run without any external download. Model weights
(ESM2-650M, Tranception Large) come from `setup.sh`.

## Architecture

| File | Role |
|------|------|
| `self_driving.py` | Main BO loop: training, conditional sampling, scoring, UCB acquisition, lab I/O, DB writes |
| `train_from_database.py` | Offline replay: re-train/score iteration-by-iteration from a full DB |
| `utils.py` | DB CRUD, model scoring, ESM embedding extraction, sequence helpers |
| `phenotype_monitor.py` | `watchdog` monitor that blocks until the lab writes a phenotype JSON |
| `precompute_embeddings.py` | ESM/MSA/Tranception embeddings → HDF5 (chunked) |
| `precompute_zero_shot.py` | Tranception zero-shot fitness predictions |
| `insilico_analysis/` | In-silico benchmark suite (see above) |
| `data_analysis/` | Plots/CSVs read directly from the result SQLite DB |

### Configuration

- `config.sh` — single source of truth for runtime parameters; auto-resolves `PRAXIS_ROOT`.
- `configs/model/PNPT_ESM2_650M_final.json` — ProteinNPT architecture (ESM2-650M, CNN head, 5 NPT layers).
- `configs/targets/self_driving_agent{1,2,3}.json` — per-agent target/loss configuration.

### Lab integration

Real-lab mode transfers sequence queries to the environment `file_reciever` over HTTP
(`LAB_UPLOAD_URL` env var) and blocks on `phenotype_monitor.py` until results return as JSON.
