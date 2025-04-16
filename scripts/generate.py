import torch
from transformers import pipeline
from memeGPT.model.model import Model
from memeGPT.tokenizer.tokenizer import text_tokenizer
import sys
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model("distilgpt2")
tokenizer = text_tokenizer("distilgpt2")

path = '/home/kandarpa-sarkar/Downloads/memeGPT2/ver1.pt'

model.load_weights(path, map_location=device)

model().eval()
text_generator = pipeline(
    "text-generation",
    model= model(),
    tokenizer=tokenizer(),
    device= device,
    truncation=True 
)

# model.config.pad_token_id = model.config.eos_token_id

prompt = f"prompt:{sys.argv[1]}\n meme:"
outputs = text_generator(
    prompt,
    max_length=32,
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