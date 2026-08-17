import csv
import json
import sys
import os

def read_sequence_file(file_path):
    """Read multiple sequences from a file and clean them."""
    try:
        with open(file_path, 'r') as file:
            sequences = [line.strip() for line in file.readlines() if line.strip()]
        
        if not sequences:
            raise ValueError("Sequence file is empty")
        return sequences
    except FileNotFoundError:
        raise FileNotFoundError(f"Sequence file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading sequence file: {e}")

def find_sequence_fragments(sequence, csv_file_path, sequence_number):
    """Find fragments for a given sequence."""
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            data = {row[0]: row[1:] for row in csv_reader}
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    except csv.Error as e:
        raise ValueError(f"Error reading CSV file: {e}")

    result = []
    start = 0
    while start < len(sequence):
        found = False
        for p in data:
            for f, fragment in enumerate(data[p]):
                if fragment and sequence.startswith(fragment, start):
                    result.append(f"{p}f{f}")
                    start += len(fragment)
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"No matching fragment found for sequence starting at position {start}")

    return result

def get_chimera_sequence(fragments):
    """Create parent sequence string from list of fragments"""
    sorted_fragments = sorted(fragments, key=lambda x: int(x[3:]))
    chimera_sequence = ''.join(fragment[1:2] for fragment in sorted_fragments)
    print("chimera sequence: ", chimera_sequence)
    return chimera_sequence

def create_plate_layout(sequences):
    """Create plate layout dictionary"""
    layout = {}
    sequence_rows = 'ABCDEF'  # one row per sequence
    sugars = ['glu', 'xyl', 'man']

    # Get parent sequences for each sequence
    parent_sequences = {
        seq_num: get_chimera_sequence(fragments)
        for seq_num, fragments in sequences.items()
    }

    # --- Place Samples --- each row = 1 sequence, cols 1-3/4-6/7-9 = glu/xyl/man reps
    for seq_idx, seq_num in enumerate(sorted(sequences.keys())):
        row = sequence_rows[seq_idx]
        for sugar_idx, sugar in enumerate(sugars):
            sugar_col_start = sugar_idx * 3 + 1
            for rep in range(1, 4):
                col = sugar_col_start + rep - 1
                well = f"{row}{col}"
                layout[well] = f"{parent_sequences[seq_num]}-{sugar}-R{rep}"

    # --- Place Negative Controls in Row G ---
    for sugar_idx, sugar in enumerate(sugars):
        sugar_col_start = sugar_idx * 3 + 1
        for rep in range(1, 4):
            col = sugar_col_start + rep - 1
            well = f"G{col}"
            layout[well] = f"N-{sugar}-R{rep}"

    return layout

def write_results(fragments, output_file):
    """Write results to JSON file."""    
    # Create plate layout
    layout = create_plate_layout(fragments)
    
    # Write JSON output to specified file
    with open(output_file, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"Plate layout written to {output_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <sequence_file> <csv_file_path> <output_json>")
        sys.exit(1)

    sequence_file = sys.argv[1]
    csv_file_path = sys.argv[2]
    output_file = sys.argv[3]

    try:
        # Read sequences
        sequences = read_sequence_file(sequence_file)
        
        # Process each sequence
        sequence_fragments = {}
        for i, sequence in enumerate(sequences, 1):
            fragments = find_sequence_fragments(sequence, csv_file_path, i)
            sequence_fragments[str(i)] = fragments
            print(f"Sequence {i} fragments: {fragments}")
        
        # Write results
        write_results(sequence_fragments, output_file)
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()