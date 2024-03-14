#!/bin/bash

# The path to your Python script
PYTHON_SCRIPT_PATH="./src/generate_summaries.py"

# Example arguments - you can modify these to match your actual file paths and column names
INPUT_FILE_PATH="./results/generic_prompt_output/test_inference.csv"
OUTPUT_FILE_PATH="./results/generic_prompt_outpu/test_inference_summarized.csv"
INPUT_COLUMN_NAME="llava_report"
OUTPUT_COLUMN_NAME="llava_rep_summary"

# Run the Python script with the specified arguments
python $PYTHON_SCRIPT_PATH --input_file $INPUT_FILE_PATH --output_file $OUTPUT_FILE_PATH --input_column $INPUT_COLUMN_NAME --output_column $OUTPUT_COLUMN_NAME
