# DSC180B-Capstone

Our capstone project is using [LLaVA](https://github.com/haotian-liu/LLaVA) as our LLM and vision encoder model. 

## Install
First follow the LLaVA install instructions [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) and download the Model weights for [LLaVA-1.5 13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) into the LLaVA base directory.

Clone this directory into the LLaVA base directory such that `LLaVA` is this repos parent directory.

## Environments
The project was ran on Ubuntu 22.04 and not tested on any other OS.

There are two Conda environments that we are using.

1. datalite_env: Environment where all data and evaluation scripts are ran

2. llava_123: Environment where LLaVA is used for training and inference

To install an environment use

```
conda env create -f data_env.yml
```

or 

```
conda env create -f llava_env.yml
```

You can also use the environment created through the LLaVA install as your environment to run model training and inferences.

Activate your conda environment using `conda activate llava_123` or `conda activate datalite_env` and deactivate using `conda deactivate`

## Basic Usage
```
make
```
will run all build targets.

Current default values in all the script files are for the generic prompt. Modify the paths in all files in `scripts/` to run on the context prompt.
## Build Targets

1. Data
2. Training
3. Inference
4. Model Evaluation
5. Classification
6. BART
7. Summary
8. Few Shot Prompts

### Data
```
make data
```
This build target will generate jpgs from hdf5s, create a train test split, and generate the json and jsonl for training and inference with LLaVA with two different prompts.

This creates 4 different files

1. `generic_prompt_train.json`: entries for generic fine-tuning LLaVA
2. `generic_prompt_test.jsonl`: entires for generic prompt inferences. 
3. `context_prompt_train.json`: entries for context prompt fine-tuning
4. `context_prompt_test.jsonl`: entries for context prompt inferences.

### Training

To run the fine-tuning on LLaVA, use
```
make train
```

### Inference

To generate answers for the data in the test set, run
```
make inference
```

### Evaluation
Generates a similarity score using a sentence transformer, generates plots by author, and generates csvs for classification.
```
make evaluation
```

### Bart Scoring
Runs BART model to score reports based off of passed in candidate labels
```
make bart
```

### BART Output Evaluation
Generates Confusion Matrices and ROC curves for the BART scoring.
```
make bart-eval
```

### Summary
Uses a transformer to generate a report summary that is used for the bart scoring.
```
make summary
```

### Few Shot Prompts
Modifies the prompts to have instructions about how to read the data.
To modify the generic prompt:
```
make few_shot TARGET=generic
```
or
To modify the context prompt
```
make few_shot TARGET=context
```