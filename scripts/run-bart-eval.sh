#!/bin/bash

# Define variables for the input paths and the output path
LLAVA_SCORES_PATH="./results/generic_prompt_output/llava_classified_reports_test_set.csv"
RADIOLOGIST_SCORES_PATH="./results/generic_prompt_output/radiologist_classified_reports_test_set.csv"
OUTPUT_PATH="./data/plots"

# Execute the Python script with the provided arguments
python ./bart_eval.py \
  --llava_scores_path "$LLAVA_SCORES_PATH" \
  --radiologist_scores_path "$RADIOLOGIST_SCORES_PATH" \
  --output_path "$OUTPUT_PATH"

echo "Script execution completed. Check the output directory for results."
