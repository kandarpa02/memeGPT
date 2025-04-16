from transformers import GPT2Tokenizer
from memeGPT.model.model import Model

class text_tokenizer(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

    def __call__(self):
        return self.tokenizer
    
    