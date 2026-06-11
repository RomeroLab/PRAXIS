#!/bin/bash

# Configuration
PYTHON_SCRIPT="generate_assay_plate.py"
SEQUENCE_FILE="sequence_query.txt"
CSV_FILE="sequence_segments.csv"
OUTPUT_JSON="plate_layout.json"

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

# Remove plate_layout.json if it exists (whether file or directory)
rm -rf "$OUTPUT_JSON"

# Run the Python script
echo "Starting plate layout generation..."
echo "Input sequence file: $SEQUENCE_FILE"
echo "Fragment data: $CSV_FILE"
echo "Output JSON: $OUTPUT_JSON"

python3 "$PYTHON_SCRIPT" "$SEQUENCE_FILE" "$CSV_FILE" "$OUTPUT_JSON" || error_exit "Failed to run Python script"

if [ -f "$OUTPUT_JSON" ]; then
    echo "Successfully generated plate layout at $OUTPUT_JSON"
else
    error_exit "Failed to generate output JSON file"
fi

echo "Process completed successfully."