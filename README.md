# PRAXIS

**PRAXIS** is a self-driving platform for protein engineering: a machine-learning **agent**
proposes enzyme variants and a robotic **environment** synthesizes and assays them, closing
the design–test loop without a human in the inner cycle.

This repository accompanies the paper *"<TODO: paper title>"* and contains both halves of the
loop plus an in-silico benchmark suite for reproducing the computational results.

```
            ┌─────────────────────────────┐
            │   agent/   (this is the ML)  │
            │  ProteinNPT surrogate + BO   │
            │  proposes next variants      │
            └───────────────┬─────────────┘
            sequence queries │   ▲ phenotypes
                             ▼   │
            ┌─────────────────────────────┐
            │ environment/  (the wet lab)  │
            │  assembly → TXTL → assay →   │
            │  plate processing            │
            └─────────────────────────────┘
```

## Repository layout

| Path | What it is |
|------|------------|
| [`agent/`](agent) | The ML/Bayesian-optimization agent (ProteinNPT surrogate, conditional sampling, UCB acquisition, lab I/O). |
| [`agent/insilico_analysis/`](agent/insilico_analysis) | In-silico benchmark suite (GFP/GB1/PAB1/UBE4 vs. published DMS oracles). |
| [`environment/`](environment) | The lab-automation system that runs the physical assays and returns phenotypes. |
| [`data/`](data) | Download script + manifest for the large artifacts hosted externally. |

The two halves run independently and communicate over HTTP/SSH, so you can run the agent and
benchmarks without any lab hardware.

## Quickstart

### 1. In-silico benchmarks (no lab required)

```bash
cd agent
conda env create -f self_driving_env.yml && conda activate self_driving_env
bash ../data/download_data.sh                       # fetch DMS oracle datasets + weights
bash insilico_analysis/run_full_benchmark.sh 0      # 4 proteins × 5 methods × 5 seeds
```

Each benchmark seeds with 10 random dataset sequences and uses the protein's DMS landscape as a
ground-truth oracle. See [`agent/README.md`](agent/README.md#in-silico-benchmarks).

### 2. The agent (closed-loop design)

```bash
cd agent
bash self_driving_single.sh        # one acquisition strategy
bash self_driving_multiple.sh      # three specificity agents in parallel
```

See [`agent/README.md`](agent/README.md).

### 3. The environment (lab automation)

```bash
cd environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp configs/lab_config.example.yml configs/lab_config.yml   # then edit
python lab_controller.py
```

See [`environment/README.md`](environment/README.md).

## Data & weights

The repository ships only code and small reference files. Large artifacts — benchmark DMS oracle
datasets, model weights (ESM2-650M, Tranception Large), precomputed embeddings, and result
databases — are hosted externally and fetched by:

```bash
bash data/download_data.sh
```

See [`data/README.md`](data/README.md) for the manifest and the archive DOI.

## Dependencies

The agent builds on [ProteinNPT](https://github.com/OATML-Markslab/ProteinNPT)
(`proteinnpt==1.5.1`, installed via the conda environment file). ESM2 and Tranception weights are
downloaded by `agent/setup.sh`.

## Citation

If you use PRAXIS, please cite:

```bibtex
@article{praxis<TODO>,
  title   = {<TODO: paper title>},
  author  = {<TODO: author list>},
  journal = {<TODO>},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
