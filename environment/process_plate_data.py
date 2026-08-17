import re
import pandas as pd
import numpy as np
import argparse
import json
import os
from scipy import stats

def parse_plate_data(file_path):
    """
    Parses the plate reader CSV file to extract fluorescein (Read 1:487,528)
    and kinetic timecourse (Read 2:325,450) data.

    Format:
      Read 1:487,528
      <blank>
      Well ID\\tName\\tWell\\tRead 1:487,528
      SPL1\\t\\tA1\\t<value>
      ...
      <blank>
      Read 2:325,450
      <blank>
      Time\\tT° ...\\tA1\\tB1\\t...
      0:00:00\\t30.0\\t<val>\\t...
    """
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    fluorescein_data = {}
    parsed_data = []

    section = None
    kinetic_well_names = []

    for line in lines:
        line = line.rstrip('\n')
        stripped = line.strip()

        if stripped == 'Read 1:487,528':
            section = 'fluor_header'
            continue
        elif stripped == 'Read 2:325,450':
            section = 'kinetic_header'
            continue

        if section == 'fluor_header':
            if stripped.startswith('Well ID'):
                section = 'fluor_data'
            continue

        if section == 'fluor_data':
            if not stripped:
                section = None
                continue
            parts = stripped.split('\t')
            # cols: Well ID, Name, Well, value
            if len(parts) >= 4:
                well = parts[2].strip()
                try:
                    fluorescein_data[well] = float(parts[3])
                except ValueError:
                    fluorescein_data[well] = np.nan

        if section == 'kinetic_header':
            if stripped.startswith('Time'):
                parts = stripped.split('\t')
                # cols: Time, T°, well1, well2, ...
                kinetic_well_names = parts[2:]
                section = 'kinetic_data'
            continue

        if section == 'kinetic_data':
            if not stripped:
                section = None
                continue
            parts = stripped.split('\t')
            if len(parts) < 2:
                continue
            # Parse time string H:MM:SS → seconds
            try:
                h, m, s = map(int, parts[0].split(':'))
                time_s = h * 3600 + m * 60 + s
            except ValueError:
                continue
            time_point_data = {'time_s': time_s}
            for i, well_name in enumerate(kinetic_well_names):
                col_idx = i + 2  # offset past Time and T°
                try:
                    time_point_data[well_name] = float(parts[col_idx])
                except (ValueError, IndexError):
                    time_point_data[well_name] = np.nan
            parsed_data.append(time_point_data)

    fluorescein_series = pd.Series(fluorescein_data, name='fluorescein')

    timecourse_df = pd.DataFrame(parsed_data)

    if timecourse_df.empty:
        return fluorescein_series, timecourse_df

    # Reorder columns to be logical and sort by time
    timecourse_df = timecourse_df.sort_values(by='time_s').reset_index(drop=True)
    time_col = timecourse_df[['time_s']]
    well_cols = sorted([col for col in timecourse_df.columns if col != 'time_s'])
    timecourse_df = pd.concat([time_col, timecourse_df[well_cols]], axis=1)

    return fluorescein_series, timecourse_df

def calculate_mean_rate(df, well_id):
    """Calculates the mean rate (slope) for a well via linear regression over all time points."""
    if well_id not in df.columns:
        return np.nan
    valid = df[['time_s', well_id]].dropna()
    if len(valid) < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(valid['time_s'], valid[well_id])
    return slope

# Minimum full-trace slope (RFU/sec) to attempt piecewise fitting.
# Based on well A1 from 20260227 run — a fast but fully linear enzyme.
PIECEWISE_SLOPE_THRESHOLD = 30.0

def calculate_initial_rate(df, well_id):
    """
    Calculates the initial reaction rate using continuous piecewise linear fitting.
    Fits y = a*t + b + c*max(0, t - t_bp) and returns the first segment slope (a).
    Falls back to full-trace linear regression if the breakpoint isn't justified (BIC).
    Only called when full-trace slope exceeds PIECEWISE_SLOPE_THRESHOLD.
    """
    if well_id not in df.columns:
        return np.nan
    valid = df[['time_s', well_id]].dropna()
    if len(valid) < 6:
        return calculate_mean_rate(df, well_id)

    t = valid['time_s'].values
    y = valid[well_id].values
    n = len(t)

    # Single-line fit for BIC comparison
    slope_full, intercept_full, _, _, _ = stats.linregress(t, y)
    sse_single = np.sum((y - (slope_full * t + intercept_full)) ** 2)
    bic_single = n * np.log(sse_single / n) + 2 * np.log(n)

    best_sse = np.inf
    best_slope1 = slope_full
    best_bp_time = None

    # Try each candidate breakpoint (min 3 points per segment)
    for i in range(3, n - 2):
        t_bp = t[i]
        X = np.column_stack([t, np.maximum(0, t - t_bp), np.ones(n)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        sse_pw = np.sum((y - X @ coeffs) ** 2)

        if sse_pw < best_sse:
            best_sse = sse_pw
            best_slope1 = coeffs[0]
            best_bp_time = t_bp

    # Use BIC to decide: piecewise (4 params) vs single line (2 params)
    bic_pw = n * np.log(best_sse / n) + 4 * np.log(n)

    if bic_pw < bic_single:
        return best_slope1
    else:
        return slope_full

def process_timecourse_data(timecourse_file, output_file, sequence_file):
    """
    Processes time-course plate reader data to calculate enzyme activity.
    """
    # 1. Read sequences from the sequence file
    try:
        with open(sequence_file, 'r') as f:
            sequences = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Sequence file not found at {sequence_file}")
        return {}

    # 2. Parse the time-course data from the CSV file
    fluorescein_series, raw_data_df = parse_plate_data(timecourse_file)

    # Initialize sequence validity, to be updated by fluorescein checks
    sequence_validity = {seq: True for seq in sequences}

    # 2.5 Apply fluorescein normalization
    norm_factors = pd.Series(dtype=float)
    if not fluorescein_series.empty:
        fluorescein_df = fluorescein_series.reset_index()
        fluorescein_df.columns = ['well', 'fluorescein']
        # Group by (row, substrate): each group is the 3 replicates for one enzyme+substrate
        col_num = fluorescein_df['well'].str[1:].astype(int)
        fluorescein_df['row'] = fluorescein_df['well'].str[0]
        fluorescein_df['substrate_group'] = (col_num - 1) // 3
        fluorescein_df['group'] = fluorescein_df['row'] + fluorescein_df['substrate_group'].astype(str)

        group_averages = fluorescein_df.groupby('group')['fluorescein'].mean()
        fluorescein_df['col_avg'] = fluorescein_df['group'].map(group_averages)
        
        # Calculate normalization factor where possible, otherwise default to 1 (no change)
        fluorescein_df['norm_factor'] = np.where(
            (fluorescein_df['col_avg'] > 0) & (fluorescein_df['fluorescein'] > 0),
            fluorescein_df['fluorescein'] / fluorescein_df['col_avg'],
            1.0
        )
        
        norm_factors = fluorescein_df.set_index('well')['norm_factor']

    if raw_data_df.empty:
        print("Could not parse any valid time-course data from the input file.")
        # Return a default empty structure if no data is parsed
        phenotype_data = {}
        for seq in sequences:
            phenotype_data[seq] = {"measurements": [0.0, 0.0, 0.0], "valid": True}
        return phenotype_data

    # 3. Define the experimental layout
    sequence_rows = ['A', 'B', 'C', 'D', 'E', 'F']
    substrate_rep_cols = {'glucose': ['1', '2', '3'], 'xylose': ['4', '5', '6'], 'mannose': ['7', '8', '9']}
    control_row = 'G'
    substrate_order = ['glucose', 'xylose', 'mannose']

    # 3.5. Perform fluorescein-based validity checks
    if not fluorescein_series.empty:
        fluorescein_well_map = fluorescein_df.set_index('well')
        all_sequences_invalid = False

        # Check control wells first
        for substrate, cols in substrate_rep_cols.items():
            control_replicate_wells = [f"{control_row}{col}" for col in cols]
            for well in control_replicate_wells:
                if well in fluorescein_well_map.index:
                    well_data = fluorescein_well_map.loc[well]
                    # Check if col_avg is not zero to avoid division errors
                    if well_data['col_avg'] > 0 and (well_data['fluorescein'] / well_data['col_avg']) < 0.25:
                        print(f"CONTROL WELL FAILED: {well} for {substrate} fluorescein is too low. Invalidating all sequences.")
                        all_sequences_invalid = True
                        break
            if all_sequences_invalid:
                break

        if all_sequences_invalid:
            for seq in sequences:
                sequence_validity[seq] = False
        else:
            # Check experimental wells if controls are okay
            for i, seq in enumerate(sequences):
                if i < len(sequence_rows):
                    row = sequence_rows[i]
                    is_sequence_valid = True
                    for substrate, cols in substrate_rep_cols.items():
                        experimental_replicate_wells = [f"{row}{col}" for col in cols]
                        for well in experimental_replicate_wells:
                            if well in fluorescein_well_map.index:
                                well_data = fluorescein_well_map.loc[well]
                                if well_data['col_avg'] > 0 and (well_data['fluorescein'] / well_data['col_avg']) < 0.25:
                                    print(f"WELL FAILED: {well} for sequence {seq} ({substrate}) fluorescein is too low. Invalidating sequence.")
                                    is_sequence_valid = False
                                    break 
                        if not is_sequence_valid:
                            break
                    sequence_validity[seq] = is_sequence_valid


    analysis_df = raw_data_df.copy()

    sequence_mapping = {}

    # 4. Process each sequence and substrate combination
    for i, seq in enumerate(sequences):
        if i >= len(sequence_rows):
            sequence_mapping[seq] = [0.0, 0.0, 0.0]
            continue

        row = sequence_rows[i]
        final_measurements = []
        for substrate in substrate_order:
            replicate_wells = [f"{row}{col}" for col in substrate_rep_cols[substrate]]
            control_wells = [f"{control_row}{col}" for col in substrate_rep_cols[substrate]]

            # Handle overflow by finding the first time point with NaN
            replicates_df = analysis_df[replicate_wells]
            overflow_at_time = replicates_df.isna().any(axis=1)
            first_overflow_idx_series = overflow_at_time[overflow_at_time].index

            if not first_overflow_idx_series.empty:
                first_overflow_idx = first_overflow_idx_series[0]
                search_df = analysis_df.loc[analysis_df.index < first_overflow_idx].copy()
            else:
                search_df = analysis_df.copy()

            # Calculate mean rates for replicates (overflow-trimmed) and controls.
            # Slopes from linregress are in RFU/sec; multiply by 1000 to convert to mRFU/sec.
            # Using mRFU/sec keeps low off-target activities (0.001–0.01 RFU/sec) well above 1,
            # ensuring ln(1+x) operates in the log regime across the full dynamic range.
            replicate_rates_mRFU = []
            control_rates_mRFU = []
            for j, rep_well in enumerate(replicate_wells):
                ctrl_well = control_wells[j]
                # Use piecewise fitting for fast enzymes to capture initial velocity
                rep_rate = calculate_mean_rate(search_df, rep_well)
                if pd.notna(rep_rate) and rep_rate > PIECEWISE_SLOPE_THRESHOLD:
                    full_rate = rep_rate
                    rep_rate = calculate_initial_rate(search_df, rep_well)
                    print(f"  {rep_well}: piecewise fit (full-trace={full_rate*1000:.1f}, initial={rep_rate*1000:.1f} mRFU/s)")
                else:
                    print(f"  {rep_well}: full-trace fit ({rep_rate*1000:.1f} mRFU/s)" if pd.notna(rep_rate) else f"  {rep_well}: no data")
                ctrl_rate = calculate_mean_rate(analysis_df, ctrl_well)

                rep_nf = norm_factors.get(rep_well, 1.0) if not norm_factors.empty else 1.0
                ctrl_nf = norm_factors.get(ctrl_well, 1.0) if not norm_factors.empty else 1.0
                replicate_rates_mRFU.append(rep_rate / rep_nf * 1000 if pd.notna(rep_rate) else np.nan)
                control_rates_mRFU.append(ctrl_rate / ctrl_nf * 1000 if pd.notna(ctrl_rate) else np.nan)

            # Calculate background-subtracted rate (in mRFU/sec)
            valid_replicate_rates_mRFU = [r for r in replicate_rates_mRFU if pd.notna(r)]
            valid_control_rates_mRFU = [c for c in control_rates_mRFU if pd.notna(c)]

            final_rate_mRFU = 0.0
            if valid_replicate_rates_mRFU and valid_control_rates_mRFU:
                median_rep_rate_mRFU = np.median(valid_replicate_rates_mRFU)
                median_ctrl_rate_mRFU = np.median(valid_control_rates_mRFU)
                final_rate_mRFU = median_rep_rate_mRFU - median_ctrl_rate_mRFU

            final_measurements.append(max(0.0, final_rate_mRFU))

        # Log-transform the final measurements (inputs are in mRFU/sec; ln(1+x) operates in log regime throughout)
        log_transformed_measurements = [np.log1p(m) for m in final_measurements]
        sequence_mapping[seq] = log_transformed_measurements
    
    # 5. Load existing phenotype data to preserve 'valid' status
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    # Create the final phenotype data structure
    phenotype_data = {}
    for seq, measurements in sequence_mapping.items():
        phenotype_data[seq] = {
            "measurements": measurements,
            "valid": existing_data.get(seq, {}).get("valid", True) and sequence_validity.get(seq, True)
        }
    
    return phenotype_data


def main():
    parser = argparse.ArgumentParser(description="Process time-course plate reader data to calculate enzyme activities.")
    parser.add_argument('--input', required=True, help="Path to the input directory containing the timecourse data CSV file.")
    parser.add_argument('--output', required=True, help="Path to the output phenotype.json file.")
    parser.add_argument('--sequence_file', required=True, help="Path to the sequence file.")

    args = parser.parse_args()

    # The controller passes a directory, so we construct the full path to the data file.
    timecourse_file_path = os.path.join(args.input, 'raw_plate_data.csv')

    phenotype_data = process_timecourse_data(timecourse_file_path, args.output, args.sequence_file)

    # Save the phenotype data
    if phenotype_data:
        with open(args.output, 'w') as f:
            json.dump(phenotype_data, f, indent=4)
            print(f"\nPhenotype file written to {args.output}")
    else:
        print("\nPhenotype data could not be generated.")

if __name__ == "__main__":
    main() 

    