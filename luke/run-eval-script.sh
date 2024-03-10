#!/bin/bash
# TODO: activate datalite environment
# for this script, so we don't forget to activate it when calling terminal

# The path to your Python script
SCRIPT_PATH="/home/llm-hackathon/LLaVA/luke_scratch/eval_script.py"

# Output path where the results will be saved
# mkdir "/home/llm-hackathon/LLaVA/luke_scratch/01-29-24-task-lora"

OUTPUT_PATH="/home/llm-hackathon/LLaVA/luke_scratch/03-04-24"
#!/bin/bash

# Directories or paths for the required arguments
TEST_INFERENCE_PATH='/home/llm-hackathon/LLaVA/raymond/030424_test_answers.jsonl'
#'/home/llm-hackathon/LLaVA/luke_scratch/01-29-24/test_inference.jsonl'

TRAIN_DATA_PATH='/home/llm-hackathon/LLaVA/raymond/030424patient_finding_impression.json'
#'/home/llm-hackathon/LLaVA/luke_scratch//01-29-24/train_patient_finding_impression.json'

# Ensure the script is executable: chmod +x run_script.sh

# Run the Python script with named arguments
python "$SCRIPT_PATH" \
--output_path "$OUTPUT_PATH" \
--test_inference_path "$TEST_INFERENCE_PATH" \
--train_data_path "$TRAIN_DATA_PATH"
