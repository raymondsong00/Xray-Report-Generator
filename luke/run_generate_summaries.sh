#!/bin/bash

# The path to your Python script
PYTHON_SCRIPT_PATH="/home/llm-hackathon/LLaVA/luke_scratch/generate_summaries.py"

# Example arguments - you can modify these to match your actual file paths and column names
INPUT_FILE_PATH="/home/llm-hackathon/LLaVA/luke_scratch/03-03-24-01/data/test_inference.csv"
OUTPUT_FILE_PATH="/home/llm-hackathon/LLaVA/luke_scratch/03-03-24-01/data/test_inference-l.csv"
INPUT_COLUMN_NAME="llava_report"
OUTPUT_COLUMN_NAME="llava_rep_summary"

# Run the Python script with the specified arguments
python $PYTHON_SCRIPT_PATH --input_file $INPUT_FILE_PATH --output_file $OUTPUT_FILE_PATH --input_column $INPUT_COLUMN_NAME --output_column $OUTPUT_COLUMN_NAME
