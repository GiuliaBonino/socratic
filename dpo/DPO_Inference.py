from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
import torch
from peft import AutoPeftModelForCausalLM

HG_MODEL_NAME = "TheBloke/OpenHermes-2-Mistral-7B-GPTQ"
HG_TOKENIZER_NAME = HG_MODEL_NAME

model_ref = AutoModelForCausalLM.from_pretrained(HG_MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True, quantization_config=GPTQConfig(bits=4, disable_exllama=True))
tokenizer = AutoTokenizer.from_pretrained(HG_TOKENIZER_NAME)

input = tokenizer("""I have dropped my phone in water. Now it is not working what should I do now?""", return_tensors="pt").to("cuda")

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    "./model",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id
)


trained_output = trained_model.generate(**input, generation_config=generation_config)
print(f"Trained Output: {tokenizer.decode(trained_output[0], skip_special_tokens=True)}\n")

ref_output = model_ref.generate(**input, generation_config=generation_config)
print(f"Original Output: {tokenizer.decode(ref_output[0], skip_special_tokens=True)}\n")
