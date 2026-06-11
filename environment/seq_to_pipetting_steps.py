import csv
import sys
import os

def read_sequence_file(file_path):
    """Read multiple sequences from a file and clean them."""
    try:
        with open(file_path, 'r') as file:
            sequences = [line.strip() for line in file.readlines() if line.strip()]
        
        print(f"Read {len(sequences)} sequences:")
        for i, seq in enumerate(sequences, 1):
            print(f"Sequence {i} (length: {len(seq)}): {seq[:50]}...")
        
        if not sequences:
            raise ValueError("Sequence file is empty")
        return sequences
    except FileNotFoundError:
        raise FileNotFoundError(f"Sequence file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading sequence file: {e}")

def find_sequence_fragments(sequence, csv_file_path, sequence_number):
    print(f"\nProcessing Sequence {sequence_number}:")
    print(f"Searching for fragments in sequence: {sequence[:50]}...")

    # Read the CSV file
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            data = {row[0]: row[1:] for row in csv_reader}
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    except csv.Error as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Initialize result which will store fragment identifiers i.e. p1f0
    result = []
    
    # initialize index which will store 96-well plate locations i.e. A4
    index = []

    # dictionary mapping fragment identifiers to a 96-well plate
    well_dict = {"p1f0": "B3", "p1f1": "B4", "p1f2": "B5", "p1f3": "B6", "p1f4": "B7", "p1f5": "B8", "p1f6": "B9", "p1f7": "B10",
                 "p2f0": "C3", "p2f1": "C4", "p2f2": "C5", "p2f3": "C6", "p2f4": "C7", "p2f5": "C8", "p2f6": "C9", "p2f7": "C10",
                 "p3f0": "D3", "p3f1": "D4", "p3f2": "D5", "p3f3": "D6", "p3f4": "D7", "p3f5": "D8", "p3f6": "D9", "p3f7": "D10",
                 "p4f0": "E3", "p4f1": "E4", "p4f2": "E5", "p4f3": "E6", "p4f4": "E7", "p4f5": "E8", "p4f6": "E9", "p4f7": "E10",
                 "p5f0": "F3", "p5f1": "F4", "p5f2": "F5", "p5f3": "F6", "p5f4": "F7", "p5f5": "F8", "p5f6": "F9", "p5f7": "F10",
                 "p6f0": "G3", "p6f1": "G4", "p6f2": "G5", "p6f3": "G6", "p6f4": "G7", "p6f5": "G8", "p6f6": "G9", "p6f7": "G10"}

    # Iterate through the sequence
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

    for i in result:
        if i in well_dict:
            index.append(well_dict[i])
        else:
            print(f"Warning: No well mapping found for fragment {i}")

    return result, index

def format_well(well):
    """Convert well format from B3 to B03"""
    letter = well[0]
    number = well[1:]
    return f"{letter}0{number}" if len(number) == 1 else f"{letter}{number}"

def get_destination_well(seq_num):
    """Get destination well based on sequence number"""
    destinations = {1: 'B01', 2: 'C01', 3: 'D01', 4: 'E01', 5: 'F01', 6: 'G01'}
    return destinations[seq_num]

def write_worklist(all_wells, output_dir):
    """Write worklist CSV file"""
    worklist_file = os.path.join(output_dir, 'fragment_assembly_worklist.csv')
    
    rows = []
    index = 1
    
    for seq_num, wells in enumerate(all_wells, 1):
        # Ensure we have exactly 7 fragments
        if len(wells) != 8:
            raise ValueError(f"Sequence {seq_num} has {len(wells)} fragments, expected 8")
        
        destination_well = get_destination_well(seq_num)
        
        for well in wells:
            rows.append({
                'Index': f"{index:02d}",
                'Source_Plate': 'DNA_frags',
                'Source_Well': format_well(well),
                'Destination_Plate': 'Rxn_plate',
                'Destination_Well': destination_well,
                'Volume': '5',
                'Pre_Aspirate_Mix_Volume': '0',
                'Post_Dispense_Mix_Volume': '0'
            })
            index += 1
    
    with open(worklist_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Index', 'Source_Plate', 'Source_Well', 'Destination_Plate',
            'Destination_Well', 'Volume', 'Pre_Aspirate_Mix_Volume',
            'Post_Dispense_Mix_Volume'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    
    print(f"- Generated worklist: {worklist_file}")



def main():
    print("Starting sequence fragment finder...")
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <sequence_file> <csv_file_path> <output_dir>")
        sys.exit(1)

    sequence_file = sys.argv[1]
    csv_file_path = sys.argv[2]
    output_dir = sys.argv[3]

    try:
        # Read sequences from file
        sequences = read_sequence_file(sequence_file)
        
        # Process each sequence
        all_fragments = []
        all_wells = []
        
        for i, sequence in enumerate(sequences, 1):
            fragments, wells = find_sequence_fragments(sequence, csv_file_path, i)
            all_fragments.append(fragments)
            all_wells.append(wells)
        
        # Write worklist
        write_worklist(all_wells, output_dir)
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()