import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilgpt2"         
adapter_path = sys.argv[1]
prompt = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    model_name,            
    device_map="auto"                  
)

model = PeftModel.from_pretrained(
    base,
    adapter_path,
    is_trainable=False    
)

model.eval().to(device)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    out_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


# python <adapter_path> <prompt>