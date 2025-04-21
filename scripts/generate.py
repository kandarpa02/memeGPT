import torch
from transformers import pipeline, GenerationConfig
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
import sys
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = sys.argv[1]
wrapper = Model(model_name)
tokenizer = text_tokenizer(model_name)()

path = sys.argv[2]
wrapper.load_weights(path, model_name)
model = wrapper.model 
model.eval()

model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.eos_token_id

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16  # Match quantization dtype
)

prompt = f"prompt:{sys.argv[3]}\n:"

with torch.inference_mode():
    outputs = text_generator(
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        do_sample=True, 
        num_return_sequences=1,
        repetition_penalty=1.1 
    )

sys.stdout.write(f"{outputs[0]['generated_text']}")
sys.stdout.flush()

# python generate.py --model_name <base_model> --weight_path <lora_adapters> --prompt "your text"