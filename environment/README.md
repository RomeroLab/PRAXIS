# PRAXIS — Environment (lab automation)

The **environment** half of PRAXIS: an automated laboratory system that executes the
physical test side of the design–test loop. It orchestrates the full pipeline —
DNA sequence input → fragment assembly → PCR → cell-free expression (TXTL) → enzyme assays →
data processing → transfer of phenotypes back to the GPU server running the
[`agent/`](../agent).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Create your config from the template (the real file is gitignored)
cp configs/lab_config.example.yml configs/lab_config.yml
# then edit configs/lab_config.yml — or export PRAXIS_GPU_* / PRAXIS_SSH_KEY env vars
```

## Running

```bash
python lab_controller.py     # main orchestrator (file-watcher state machine)
python lab_cam.py            # camera streaming web interface
python file_reciever.py      # file-upload API the agent posts sequence queries to (port 8000)

# DNA inventory management
python manage_dna.py --report
python manage_dna.py --refill
```

## Architecture

### State machine
`lab_controller.py` drives a `LabState` machine:
`IDLE → SEQUENCE_RECEIVED → MONITORING_EVAGREEN → EXPERIMENTING → PROCESSING → IDLE`,
with transitions triggered by filesystem events via `watchdog`.

### File-watcher pipeline
1. **SequenceHandler** — `data/sequence_query.txt` → `seq_to_pipetting_steps.py` + `generate_assay_plate.py` produce robot worklists and the plate layout.
2. **EvagreenHandler** — `data/raw_evagreen_data/…csv` → `update_valid_assemblies.py` filters samples failing qPCR QC.
3. **PlateDataHandler** — `data/raw_plate_data/…csv` → `process_plate_data.py`, archives raw data, updates tracking, and SFTP-transfers `phenotype.json` to the agent's GPU server.

### Key modules

| File | Role |
|------|------|
| `lab_controller.py` | Orchestrator, state machine, inventory tracking |
| `process_plate_data.py` | Parses plate-reader CSV; statistical normalization + Mann-Whitney analysis → `phenotype.json` |
| `seq_to_pipetting_steps.py` | Maps sequences to the fragment library; emits 6 robot worklist CSVs |
| `generate_assay_plate.py` | Assigns sequences to 96-well plate positions |
| `update_valid_assemblies.py` | Removes samples failing the Evagreen dsDNA check |
| `data_transfer.py` | Paramiko SFTP client to the agent's GPU server |
| `manage_dna.py` | CLI for DNA fragment inventory |
| `notify.py` | Push notifications via Pushover |

### Configuration

`configs/lab_config.yml` (copied from `configs/lab_config.example.yml`) holds paths, filenames,
SFTP target, and Pushover notification settings. SFTP and notification values can be overridden
by `PRAXIS_GPU_HOST`, `PRAXIS_GPU_USER`, `PRAXIS_GPU_REMOTE_PATH`, `PRAXIS_SSH_KEY`,
`PRAXIS_GPU_PORT` environment variables.

`examples/` contains a sample plate-reader export and assay output for reference.
