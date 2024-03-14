#!/bin/bash

# The path to your Python script
SCRIPT_PATH="./src/eval_script.py"

OUTPUT_PATH="./results/generic_prompt_output"

mkdir -p $OUTPUT_PATH

# Directories or paths for the required arguments
TEST_INFERENCE_PATH='./data/generic_prompt_answers.jsonl'

TRAIN_DATA_PATH='./data/generic_prompt_train.json'

TEST_INFERENCE_ANSWERS_PATH='./data/test_set_radiologist_reports.csv'

# Ensure the script is executable: chmod +x run_script.sh

# Run the Python script with named arguments
python "$SCRIPT_PATH" \
--output_path "$OUTPUT_PATH" \
--test_inference_path "$TEST_INFERENCE_PATH" \
--train_data_path "$TRAIN_DATA_PATH" \
--test_inference_answers_path "$TEST_INFERENCE_ANSWERS_PATH"
