#!/bin/bash

# Replace the following placeholders with the actual paths and file names.
CHECKPOINTS_PATH="../checkpoints/llava-v1.5-13b-generic-prompt"
IMAGE_FOLDER="./data/jpg"
QUESTION_FILE="./data/generic_prompt_test.jsonl"
ANSWER_PATH="./data/generic_prompt_answers.jsonl"
MODEL_BASE="../llava-v1.5-13b"

# Now run the model evaluation
python ../llava/eval/model_vqa.py \
  --model-path $CHECKPOINTS_PATH \
  --image-folder $IMAGE_FOLDER \
  --question-file $QUESTION_FILE \
  --answers-file $ANSWER_PATH \
  --model-base $MODEL_BASE
