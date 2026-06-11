# PRAXIS — Agent (generative protein language model)

The **agent** in the autonomous-experimental-interaction loop: a generative protein language
model, based on [ProteinNPT](https://github.com/OATML-Markslab/ProteinNPT), that learns
relationships between sequence variation and function from experimental data while leveraging
evolutionary priors from pretrained sequence models. The agent proposes informative experiments;
the [`environment/`](../environment) laboratory executes them and returns measurements; those
measurements update the agent's internal model and guide the next round of design.

Each round:

1. Update the model on all accumulated sequence–function measurements (stored in SQLite),
   progressively refining its representation of the local fitness landscape.
2. Use high-performing sequences from previous rounds as starting points for diversification,
   generating candidate variants via conditional sampling.
3. Evaluate candidates by predicted function and uncertainty, and select top-ranked variants
   using upper confidence bound (UCB) acquisition.
4. Send the selected designs to the laboratory ([`environment/`](../environment)) over HTTP/SSH.
5. Incorporate the returned measurements and repeat.

## Setup

```bash
# 1. Create the conda environment (installs proteinnpt==1.5.1 and all deps via pip)
conda env create -f self_driving_env.yml
conda activate self_driving_env

# 2. Point PRAXIS_ROOT at this directory (scripts auto-resolve it, but you can set it explicitly)
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

Configure `config.sh` first (`run_mode`, `run_id`, optimization hyperparameters). `PRAXIS_ROOT`
auto-resolves to this directory; everything else is derived from it.

```bash
bash self_driving_single.sh      # a single agent / objective
bash self_driving_multiple.sh    # the multi-agent campaign: three agents optimizing
                                 # glucose / xylose / mannose specificity, sharing one
                                 # experimental memory and learning model
```

Replay model updating offline from an existing experimental database:

```bash
bash train_from_database_multiple.sh
```

## In-silico benchmarks

Before coupling the agent to the laboratory, its ability to select informative experiments is
validated using existing deep mutational scanning (DMS) datasets as in-silico experimental
environments: the measured sequence–function data serves as a ground-truth oracle, and the agent
runs simulated rounds of experimental design under controlled conditions.

The agent is challenged to optimize fitness across four datasets spanning diverse structures,
functions, and degrees of landscape epistasis — **Pab1, GB1, Ube4b, and avGFP** (`pab1`, `gb1`,
`ube4`, `gfp`). Each search starts from a **10-sequence random seed** (round 1) and then runs
**19 acquisition rounds of 10 sequences each** (190 acquired; 200 measured in total), comparing
**5 search strategies**:

| method (code) | strategy |
|---|---|
| `full_pipeline` | ProteinNPT + UCB (uncertainty-guided) |
| `greedy` | ProteinNPT + greedy (highest-scoring only, ignores uncertainty) |
| `onehot_mlp` | one-hot MLP + UCB |
| `onehot_mlp_greedy` | one-hot MLP + greedy |
| `random` | random sampling |

```bash
# All datasets × strategies × seeds on GPU 0 (completed runs auto-skip)
bash insilico_analysis/run_full_benchmark.sh 0

# One dataset / one strategy
bash insilico_analysis/run_full_benchmark.sh 0 gfp
bash insilico_analysis/run_full_benchmark.sh 1 gb1 greedy

# Analyze + plot the sample-efficiency (fitness-trajectory) figure
python insilico_analysis/benchmark_analysis.py
python insilico_analysis/plot_paper_figures.py
```

The oracle datasets (`insilico_analysis/dataset_oracle/<protein>/<protein>_SeqFxnDataset.pkl`)
are bundled in the repository, so the benchmarks run without any external download. Model weights
(ESM2-650M, Tranception Large) come from `setup.sh`.

## Architecture

| File | Role |
|------|------|
| `self_driving.py` | Main optimization loop: model updating, conditional sampling, scoring, UCB acquisition, laboratory I/O, DB writes |
| `train_from_database.py` | Offline replay: re-train/score iteration-by-iteration from a full experimental database |
| `utils.py` | DB CRUD, model scoring, ESM embedding extraction, sequence helpers |
| `phenotype_monitor.py` | `watchdog` monitor that blocks until the laboratory writes a measurement JSON |
| `precompute_embeddings.py` | ESM/MSA/Tranception embeddings → HDF5 (chunked) |
| `precompute_zero_shot.py` | Tranception zero-shot fitness predictions |
| `insilico_analysis/` | In-silico benchmark suite (see above) |

### Configuration

- `config.sh` — single source of truth for runtime parameters; auto-resolves `PRAXIS_ROOT`.
- `configs/model/PNPT_ESM2_650M_final.json` — ProteinNPT architecture (ESM2-650M, CNN head, 5 NPT layers).
- `configs/targets/self_driving_agent{1,2,3}.json` — per-agent target/loss configuration for the
  multi-agent campaign (glucose / xylose / mannose).

### Laboratory integration

In real-laboratory mode the agent transfers sequence designs to the environment `file_reciever`
over HTTP (`LAB_UPLOAD_URL` env var) and blocks on `phenotype_monitor.py` until measurements
return as JSON.
