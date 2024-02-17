# Socratic_RLHF

Project to train Llama2 using RLHF for socratic learning

## Dataset

The dataset has been generated using a combination of QA and Chain-of-Thought datasets using Llama.cpp to generate teacher-student conversations

Scripts used:
- copy_dataset.py <dataset_name> <output_folder> -> Copies a dataset (MathQA) and splits it into folders formatting the fields
- process_dataset.sh <> -> Uses Llama.cpp to generate teacher-student conversations
- postprocess_dataset.py <dataset_name> <output_file> -> Appends all file into a csv dataset splitting the conversation into a number (NUM_SPLITS) of splits
- add_bad_answers.pynb -> Use to generate the bad answers for the RLHF process

## Model Training Pipeline

The pipline realized the Direct Policy Optimization technique for fine-tuning LLM's directly through an annotated dataset

- DPO_Trainer_Quant.py -> Trains specified quantized model using DPO [Python 3.9 only, HG Dataset]
- DPO_Trainer_Unquant.py -> Trains specified unquantized model using DPO [> Python 3.10, HG Dataset]
- DPO_Trainer_Soc.py -> Trains specified quantized model using DPO [Python 3.9 only, Socratic Dataset (only good answers)]
- DPO_Inference.py -> Used for inference, comparing with the saved model and loaded model
- DPO_Mistral.ipynb -> Trains a quantized model in a notebook 
