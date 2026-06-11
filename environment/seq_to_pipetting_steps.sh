#!/bin/bash

# Configuration
PYTHON_SCRIPT="seq_to_pipetting_steps.py"
CSV_FILE="sequence_segments.csv"
OUTPUT_DIR="output"
SEQUENCE_FILE="sequence_query.txt"

# Function to display error and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if sequence file exists
if [ ! -f "$SEQUENCE_FILE" ]; then
    error_exit "Sequence file '$SEQUENCE_FILE' not found"
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    error_exit "Python script '$PYTHON_SCRIPT' not found"
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    error_exit "CSV file '$CSV_FILE' not found"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Running Python script..."
python3 "$PYTHON_SCRIPT" "$SEQUENCE_FILE" "$CSV_FILE" "$OUTPUT_DIR" || error_exit "Failed to run Python script"

echo "Script execution completed. Results are in the '$OUTPUT_DIR' directory."