import pandas as pd
import numpy as np
import json
import os

# Constants
THRESHOLD = 10000
ORIGINAL_TXTL_PATH = 'worklists/TXTL_Wklist.csv'
NEW_TXTL_PATH = 'worklists/TXTL_Wklist_filtered.csv'
SUBSTRATE_BATCHES = [
    ('worklists/Assay_substrate_batch_glu_4.csv', 'worklists/Assay_substrate_batch_glu_4_filtered.csv'),
    ('worklists/Assay_substrate_batch_glu_3.csv', 'worklists/Assay_substrate_batch_glu_3_filtered.csv'),
    ('worklists/Assay_substrate_batch_xyl_4.csv', 'worklists/Assay_substrate_batch_xyl_4_filtered.csv'),
    ('worklists/Assay_substrate_batch_xyl_3.csv', 'worklists/Assay_substrate_batch_xyl_3_filtered.csv'),
    ('worklists/Assay_substrate_batch_man_4.csv', 'worklists/Assay_substrate_batch_man_4_filtered.csv'),
    ('worklists/Assay_substrate_batch_man_3.csv', 'worklists/Assay_substrate_batch_man_3_filtered.csv'),
]
ORIGINAL_PROTEIN_PATH = 'worklists/Assay_protein.csv'
NEW_PROTEIN_PATH = 'worklists/Assay_protein_filtered.csv'
EVAGREEN_PATH = os.path.join('data', 'raw_evagreen_data', 'raw_evagreen_data.csv')
PHENOTYPE_PATH = 'data/phenotype.json'

# Parse the EvaGreen CSV: tab-separated, header row starts with "Well"
evagreen_data = {}
in_data = False
with open(EVAGREEN_PATH, 'r', encoding='latin-1') as f:
    for line in f:
        stripped = line.strip()
        if stripped.startswith('Well\t'):
            in_data = True
            continue
        if not in_data or not stripped:
            continue
        parts = stripped.split('\t')
        if len(parts) < 1:
            continue
        well = parts[0].strip()
        try:
            evagreen_data[well] = float(parts[1]) if len(parts) > 1 and parts[1].strip() else 0.0
        except ValueError:
            evagreen_data[well] = 0.0

# Get values from A12–F12 (one EvaGreen dsDNA well per sequence)
values = {
    'A12': evagreen_data.get('A12', 0),
    'B12': evagreen_data.get('B12', 0),
    'C12': evagreen_data.get('C12', 0),
    'D12': evagreen_data.get('D12', 0),
    'E12': evagreen_data.get('E12', 0),
    'F12': evagreen_data.get('F12', 0),
}

print(f"Detected values: {values}")  # Debug print

# Update phenotype.json
with open(PHENOTYPE_PATH, 'r') as f:
    phenotype_data = json.load(f)

# Get list of sequences in order
sequences = list(phenotype_data.keys())

# Initialize or update the data structure
for seq in sequences:
    if isinstance(phenotype_data[seq], dict):
        if 'measurements' not in phenotype_data[seq]:
            phenotype_data[seq]['measurements'] = [0.0, 0.0, 0.0]
        phenotype_data[seq]['valid'] = True  # Default to True
    else:
        phenotype_data[seq] = {
            'measurements': [0.0, 0.0, 0.0],
            'valid': True
        }

# Update valid field based on evagreen values (A12=seq1 ... F12=seq6)
well_seq_order = ['A12', 'B12', 'C12', 'D12', 'E12', 'F12']
for idx, well in enumerate(well_seq_order):
    if values[well] < THRESHOLD and idx < len(sequences):
        phenotype_data[sequences[idx]]['valid'] = False

# Save updated phenotype data
with open(PHENOTYPE_PATH, 'w') as f:
    json.dump(phenotype_data, f, indent=4)

# Create mapping of which indices/wells to remove based on well positions
# TXTL_Wklist rows: 1-6=PCR transfers, 7=neg ctrl water, 8-13=extract, 14=extract ctrl, 15-20=master mix, 21=mm ctrl
txtl_remove_mapping = {
    'A12': [1, 8, 15],
    'B12': [2, 9, 16],
    'C12': [3, 10, 17],
    'D12': [4, 11, 18],
    'E12': [5, 12, 19],
    'F12': [6, 13, 20],
}

# Each sequence occupies its own row on the assay plate (A=seq1 ... F=seq6)
substrate_remove_mapping = {
    'A12': 'A',
    'B12': 'B',
    'C12': 'C',
    'D12': 'D',
    'E12': 'E',
    'F12': 'F',
}

# Determine which values to remove
txtl_indices_to_remove = []
substrate_rows_to_remove = []
for well, value in values.items():
    if value < THRESHOLD:
        print(f"Well {well} value {value} is below threshold {THRESHOLD}")
        txtl_indices_to_remove.extend(txtl_remove_mapping[well])
        substrate_rows_to_remove.append(substrate_remove_mapping[well])

# Process TXTL worklist
df_txtl = pd.read_csv(ORIGINAL_TXTL_PATH)
df_txtl = df_txtl[~df_txtl['Index'].isin(txtl_indices_to_remove)]
new_indices = [f"{i+1:02d}" for i in range(len(df_txtl))]
df_txtl['Index'] = new_indices
df_txtl.to_csv(NEW_TXTL_PATH, index=False)

# Process Substrate worklists (per-sugar batch files)
for orig, filt in SUBSTRATE_BATCHES:
    df_substrate = pd.read_csv(orig)
    df_substrate = df_substrate[~df_substrate['Destination_Well'].str[0].isin(substrate_rows_to_remove)]
    df_substrate['Index'] = [f"{i+1:02d}" for i in range(len(df_substrate))]
    df_substrate.to_csv(filt, index=False)

# Process Protein worklist
df_protein = pd.read_csv(ORIGINAL_PROTEIN_PATH)
df_protein = df_protein[~df_protein['Destination_Well'].str[0].isin(substrate_rows_to_remove)]
new_indices = [f"{i+1:02d}" for i in range(len(df_protein))]
df_protein['Index'] = new_indices
df_protein.to_csv(NEW_PROTEIN_PATH, index=False)

print(f"TXTL indices removed: {txtl_indices_to_remove}")
print(f"Rows removed from substrate and protein: {substrate_rows_to_remove}")
