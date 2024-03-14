SHELL=/bin/bash

all: data train inference evaluation classification bart

data: 
	conda deactivate
	conda activate datalite_env
	python3 ./src/data.py

train: 
	conda deactivate
	conda activate llava_123
	./scripts/finetune_task_lora_example.sh

inference: 
	conda deactivate
	conda activate llava_123
	./scripts/inference-vqa.sh

evaluation: 
	conda deactivate
	conda activate datalite_env
	./scripts/run-eval-script.sh

bart: 
	conda deactivate
	conda activate datalite_env
	./scripts/run-classification-lreport.sh
	./scripts/run-classification-rreport.sh

bart-eval: 
	conda deactivate
	conda activate datalite_env
	./scripts/run-bart-eval.sh

summary: 
	conda deactivate
	conda activate datalite_env
	./scripts/run_generate_summaries.sh

few_shot:
	@TARGET="${TARGET}"
	conda deactivate
	conda activate datalite_env
	python3 ./src/few_shot_prompts.py ${TARGET}