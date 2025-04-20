import torch
from transformers import pipeline
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
import sys
import warnings
warnings.filterwarnings("ignore")
import yaml
from peft import LoraConfig


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = sys.argv[1]
wrapper = Model(model_name)
tokenizer = text_tokenizer("gpt2")

with open(sys.argv[3], 'r') as f:
    config = yaml.safe_load(f)

peft_config = LoraConfig(
    task_type=config['peft']['task_type'],
    r=config['peft']['r'],
    lora_alpha=config['peft']['lora_alpha'],
    lora_dropout=config['peft']['lora_dropout'],
    target_modules=config['peft']['target_modules'],
    bias=config['peft']['bias']
)

path = sys.argv[2]
wrapper.load_weights(path, model_name, peft_config, map_location=device)
model = wrapper()
model.eval()
text_generator = pipeline(
    "text-generation",
    model= model,
    tokenizer=tokenizer(),
    truncation=True 
)

# model.config.pad_token_id = model.config.eos_token_id

prompt = f"prompt:{sys.argv[4]}\n:"
outputs = text_generator(
    prompt,
    max_length=512,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1
)

sys.stdout.write(
    f"{outputs[0]['generated_text']}"
)

sys.stdout.flush()

# args> --model_name --weight_path --peft_config --prompt