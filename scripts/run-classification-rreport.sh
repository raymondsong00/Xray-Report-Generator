#!/bin/bash

# Path to your Python script
SCRIPT="./src/classify_reports.py"

# Path to the input CSV file
INPUT_PATH="./results/generic_prompt_output/test_inference.csv"

# Path to the output CSV file
OUTPUT_PATH="./results/generic_prompt_output/radiologist_classified_reports_test_set.csv"

COLUMN_NAME="radiologist_report"

# Define candidate labels (adjust these labels as per your requirements)
CANDIDATE_LABELS=("pneumothorax" "pneumonia" "pleural effusion" "cardiomegaly" "edema" "rib fracture")
# Activate your Python environment if needed
# source /path/to/your/environment/bin/activate

# Run the Python script with the specified arguments
python $SCRIPT $INPUT_PATH $OUTPUT_PATH $COLUMN_NAME --candidate_labels "${CANDIDATE_LABELS[@]}"

echo "Classification script has been executed."
