from transformers import GPT2LMHeadModel
import torch

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def __call__(self):
        return self.model
    
    def model_info(self):
        print(self.model)

    def params(self):
        return self.model.parameters()
    
    def freeze(self, num = 0, ln_= 0, wte = 0, wpe = 0):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.transformer.ln_f.parameters():
            if ln_ == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.model.transformer.wte.parameters():
            if wte == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.model.transformer.wpe.parameters():
            if wpe == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for block in self.model.transformer.h[num:]:
            for param in block.parameters():
                param.requires_grad = True
    
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def load_weights(self, path, map_location = 'cuda'):
        try:
            state_dict = torch.load(path, map_location = map_location)
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean_state_dict)

        except Exception as e:
            print(f"Failed to load weights {e}")
    
