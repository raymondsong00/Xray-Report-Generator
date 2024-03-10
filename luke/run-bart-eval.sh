#!/bin/bash

# Define variables for the input paths and the output path
LLAVA_SCORES_PATH="./03-03-24-01/data/llava_classified_reports_test_set.csv"
RADIOLOGIST_SCORES_PATH="./03-03-24-01/data/radiologist_classified_reports_test_set.csv"
OUTPUT_PATH="./03-03-24-01/plots"

# Execute the Python script with the provided arguments
python ./bart_eval.py \
  --llava_scores_path "$LLAVA_SCORES_PATH" \
  --radiologist_scores_path "$RADIOLOGIST_SCORES_PATH" \
  --output_path "$OUTPUT_PATH"

echo "Script execution completed. Check the output directory for results."
