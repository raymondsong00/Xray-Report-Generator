#!/bin/bash

# Path to your Python script
SCRIPT="./classify_reports.py"

# Path to the input CSV file
INPUT_PATH="./03-04-24/data/test_inference.csv"

# Path to the output CSV file
OUTPUT_PATH="./03-03-24/data/llava_classified_reports_test_set-3.csv"

COLUMN_NAME="llava_report"

# Define candidate labels (adjust these labels as per your requirements)
CANDIDATE_LABELS=("pneumothorax" "pneumonia" "pleural effusion" "cardiomegaly" "edema" "rib fracture" "no abnormalities mentioned")

# Activate your Python environment if needed
# source /path/to/your/environment/bin/activate

# Run the Python script with the specified arguments
python $SCRIPT $INPUT_PATH $OUTPUT_PATH $COLUMN_NAME --candidate_labels "${CANDIDATE_LABELS[@]}"

echo "Classification script has been executed."
