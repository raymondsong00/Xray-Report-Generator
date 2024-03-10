#!/bin/bash

# Replace the following placeholders with the actual paths and file names.
MODEL_PATH="/home/llm-hackathon/LLaVA"
CHECKPOINTS_PATH="/home/llm-hackathon/LLaVA/checkpoints"
ANSWER_PATH="/home/llm-hackathon/LLaVA/raymond/030824_test_answers.jsonl"
#INFERENCE_FILE="/path/to/your/inference.json"
IMAGE_FOLDER="/data/UCSD_cxr/jpg"
QUESTION_FILE="/home/llm-hackathon/LLaVA/raymond/030824_test_patient_finding_impression.jsonl"

# Now run the model evaluation
python /home/llm-hackathon/LLaVA/llava/eval/model_vqa.py \
  --model-path $CHECKPOINTS_PATH/llava-v1.5-13b-03-04-24-task-lora \
  --image-folder $IMAGE_FOLDER \
  --question-file $QUESTION_FILE \
  --answers-file $ANSWER_PATH \
  --model-base $MODEL_PATH/llava-v1.5-13b
