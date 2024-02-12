#--------- IMPORTS --------#
import torch
from datasets import Dataset, load_dataset
#from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, GPTQConfig
from trl import DPOTrainer
import time
import huggingface_hub as hg

hg.login(token="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")

#------ PARAMETERS --------#
HG_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HG_MODEL_NAME = "TheBloke/OpenHermes-2-Mistral-7B-GPTQ"
HG_MODEL_NAME = "microsoft/phi-2"
HG_TOKENIZER_NAME = HG_MODEL_NAME
HG_DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
TOKEN = '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'


#------ DATASET AND PREPROCESSING ---------#
def hg_data(hg_dataset_name, split, token):
    dataset = load_dataset(
        hg_dataset_name,
        split = split,
        token = token
    )

    original_columns = dataset.column_names

    dataset = dataset.map(
        lambda sample: {
          "prompt": [prompt for prompt in sample["prompt"]],
          "chosen": sample["chosen"],
          "rejected": sample["rejected"],
        },
        batched=True,
        remove_columns=original_columns,
    )

    train_df = dataset.to_pandas().dropna()

    train_df["chosen"] = train_df["chosen"].str.get(1).str.get("content")
    train_df["rejected"] = train_df["rejected"].str.get(1).str.get("content")

    val_df = train_df.sample(10)

    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    return train_data, val_data

train_data, val_data = hg_data(HG_DATASET_NAME, "test_prefs", TOKEN)


#------ MODEL AND TOKENIZER ------#
model = AutoModelForCausalLM.from_pretrained(HG_MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)

model_ref = AutoModelForCausalLM.from_pretrained(HG_MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(HG_TOKENIZER_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#------- PEFT CONFIG ----------#
#peft_config = LoraConfig(
#        r=8,
#        lora_alpha=8,
#        lora_dropout=0.1,
#        target_modules=["q_proj", "v_proj"],
#        bias="none",
#        task_type="CAUSAL_LM",
#    )
#peft_config.inference_mode = False

#model = prepare_model_for_kbit_training(model)
#model.config.use_cache=False
#model.gradient_checkpointing_enable()
#model.config.pretraining_tp=1
#model = get_peft_model(model, peft_config)

#--------- TRAINING -----------#
training_args = TrainingArguments(
        per_device_train_batch_size=1,
        max_steps=50,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        output_dir="mistral_test",
        optim="paged_adamw_32bit",
        warmup_steps=2,
        fp16=True,
        push_to_hub=True
    )

dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_length=512,
        max_target_length=256,
        max_prompt_length=256
    )

dpo_trainer.train()







print("Hello")
