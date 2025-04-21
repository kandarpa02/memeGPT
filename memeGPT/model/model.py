from transformers import AutoModelForCausalLM
import torch
import os
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

class Model:
    def __init__(self, model_name):
        self.model_name = model_name

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = prepare_model_for_kbit_training(self.model)

    def __call__(self):
        return self.model

    def model_info(self):
        print(self.model)

    def params(self):
        return self.model.parameters()

    def freeze(self, num=0, ln_=0, wte=0, wpe=0):

        def safe_requires_grad(param, flag):
            if param.dtype.is_floating_point or param.is_complex():
                param.requires_grad = flag

        for param in self.model.parameters():
            safe_requires_grad(param, False)

        for param in self.model.transformer.ln_f.parameters():
            safe_requires_grad(param, ln_ == 1)

        for param in self.model.transformer.wte.parameters():
            safe_requires_grad(param, wte == 1)

        for param in self.model.transformer.wpe.parameters():
            safe_requires_grad(param, wpe == 1)

        for block in self.model.transformer.h[num:]:
            for param in block.parameters():
                safe_requires_grad(param, True)


    def lora(self, r=8, alpha_l=16, dropout=0.05, bias="none", target_modules= ["c_attn", "c_proj"]):
        try:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=alpha_l,
                lora_dropout=dropout,
                bias=bias,
                target_modules=target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
            print(" QLoRA adapters injected.")
            self.model.print_trainable_parameters()

        except Exception as e:
            print(f"Failed to apply QLoRA: {e}")

    def num_params(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def load_weights(self, path, base_model_name, map_location='cuda'):
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            model = PeftModel.from_pretrained(base_model, path, device_map=map_location)

            model.eval()
            self.model = model

            print(f"Successfully loaded LoRA adapter from: {path}")

        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")

        return self

