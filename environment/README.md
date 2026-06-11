# PRAXIS — Environment (autonomous robotic laboratory)

The **experimental environment** in the autonomous-experimental-interaction loop: the software
that operates a fully autonomous robotic laboratory in which the [`agent/`](../agent) acts. The
laboratory receives sequence designs from the agent, constructs the corresponding enzyme variants
through an automated DNA assembly workflow, expresses the genes in a cell-free protein expression
system, and characterizes them with fluorescence-based enzyme assays in a multimode plate reader —
then returns the measurements to the agent to guide the next round of design.

Physically, the system couples a centralized robotic arm to a liquid handler, refrigerator,
thermal cycler, plate reader, and plate stack, with all DNA fragments and reagents stored on-deck,
allowing it to progress from sequence design to functional measurement in approximately 10 hours
and to operate continuously over multi-week campaigns. This package is the control software that
coordinates experiment execution, data management, and communication with the agent.

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
python file_reciever.py      # file-upload API the agent posts sequence designs to (port 8000)

# DNA inventory management
python manage_dna.py --report
python manage_dna.py --refill
```

## Architecture

### State machine
`lab_controller.py` drives a `LabState` machine:
`IDLE → SEQUENCE_RECEIVED → MONITORING_EVAGREEN → EXPERIMENTING → PROCESSING → IDLE`,
with transitions triggered by filesystem events via `watchdog`. Robust error handling lets the
system continue operating despite occasional assembly failures or assay-quality issues.

### File-watcher pipeline
1. **SequenceHandler** — `data/sequence_query.txt` (designs from the agent) → `seq_to_pipetting_steps.py` + `generate_assay_plate.py` produce robot worklists and the plate layout.
2. **EvagreenHandler** — `data/raw_evagreen_data/…csv` → `update_valid_assemblies.py` filters assemblies failing the EvaGreen dsDNA quantitation check.
3. **PlateDataHandler** — `data/raw_plate_data/…csv` → `process_plate_data.py`, archives raw data, updates tracking, and SFTP-transfers the processed measurements (`phenotype.json`) back to the agent.

### Key modules

| File | Role |
|------|------|
| `lab_controller.py` | Orchestrator, state machine, inventory tracking |
| `process_plate_data.py` | Parses the plate-reader time-course; estimates initial reaction rates by linear (and piecewise-linear) regression of fluorescence vs. time, applies fluorescein normalization, and aggregates the substrate replicates → `phenotype.json` |
| `seq_to_pipetting_steps.py` | Maps designed sequences to the fragment library; emits 6 robot worklist CSVs |
| `generate_assay_plate.py` | Assigns sequences to 96-well plate positions |
| `update_valid_assemblies.py` | Removes assemblies failing the Evagreen dsDNA check |
| `data_transfer.py` | Paramiko SFTP client that returns measurements to the agent |
| `manage_dna.py` | CLI for DNA fragment inventory |
| `notify.py` | Push notifications via Pushover |

### Configuration

`configs/lab_config.yml` (copied from `configs/lab_config.example.yml`) holds paths, filenames,
SFTP target, and Pushover notification settings. SFTP and notification values can be overridden
by `PRAXIS_GPU_HOST`, `PRAXIS_GPU_USER`, `PRAXIS_GPU_REMOTE_PATH`, `PRAXIS_SSH_KEY`,
`PRAXIS_GPU_PORT` environment variables.

`examples/` contains a sample plate-reader export and assay output for reference.
