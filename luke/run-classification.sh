#!/bin/bash

# Path to your Python script
SCRIPT="classify_reports.py"

# Path to the input CSV file
INPUT_PATH="./test_inference.csv"

# Path to the output CSV file
OUTPUT_PATH="./classified_reports.csv"

# Define candidate labels (adjust these labels as per your requirements)
CANDIDATE_LABELS=("No Abnormality" "Aortic enlargement" "Atelectasis" "Calcification" "Cardiomegaly" "Consolidation" "ILD" "Infiltration"
"Lung Opacity" "Nodule/Mass" "Other lesion" "Pleural Effusion" "Pleural Thickening" "Pneumothorax" "Pulmonary fibrosis")

# Activate your Python environment if needed
# source /path/to/your/environment/bin/activate

# Run the Python script with the specified arguments
python $SCRIPT $INPUT_PATH $OUTPUT_PATH --candidate_labels "${CANDIDATE_LABELS[@]}"

echo "Classification script has been executed."
