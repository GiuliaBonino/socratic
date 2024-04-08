# Socratic_RLHF

Project to train Llama2 using RLHF for socratic learning

## Dataset

The dataset has been generated using a combination of QA and Chain-of-Thought datasets using Llama.cpp to generate teacher-student conversations

Scripts used:
- copy_dataset.py <dataset_name> <output_folder> -> Copies a dataset (MathQA) and splits it into files (file1, file2, ...) inside output_folder, formatting the fields as required (the source code is adapted for the MathQA dataset and must be changed to correspond and format the required fields).
- process_dataset.sh <> -> Uses Llama.cpp to generate teacher-student conversations. All the model and folder paths are declared at the top and can be change as required.
- postprocess_dataset.py <conversation_folder> <output_file> -> Appends all files in conversation_folder into a csv dataset. It generates (NUM_SPLITS) rows from each conversation adding the beginning of the conversation into the "Prompt" column and the teacher's response in the "Good answer" column.
- add_bad_answers.pynb -> Python notebook used to generate the bad answers for the RLHF process using gpt2.
  
![dataset pipeline](https://github.com/elalber2000/Socratic_RLHF/blob/main/dataset_process.PNG)

## Model Training Pipeline

The pipline realized the Direct Policy Optimization technique for fine-tuning LLM's directly through an annotated dataset

- DPO_Trainer_Quant.py -> Trains specified quantized model using DPO [Python 3.9 only, HG Dataset]
- DPO_Trainer_Unquant.py -> Trains specified unquantized model using DPO [> Python 3.10, HG Dataset]
- DPO_Trainer_Soc.py -> Trains specified quantized model using DPO [Python 3.9 only, Socratic Dataset (only good answers)]
- DPO_Inference.py -> Used for inference, comparing with the saved model and loaded model
- MistralColab.ipynb -> Trains a quantized model in a notebook
- final_model.zip -> Trained model on 5000 examples, for 30 steps

## How to run
Data generation:
 - run script copy_dataset: !python copy_dataset.py math_qa path_to_output_folder