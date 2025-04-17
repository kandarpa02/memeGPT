from torch.utils.data import DataLoader
from datasets import Dataset 
import pandas as pd
import re

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', '', text)  
    text = ' '.join(text.split())
    return text

class Load_data:
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.df = pd.read_csv(self.path)
        self.text = self.df['text'].tolist() 

    def dataloader(self, max_len=126, batch_size=8, num_workers=4):
        cleaned_text = [preprocess(text) for text in self.text]
        dataset = Dataset.from_dict({"text": cleaned_text})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def tokenize_(txt):
            tokens = self.tokenizer(
                txt['text'],
                padding='max_length',
                truncation=True,
                max_length=max_len,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        tokenized_dataset = dataset.map(tokenize_, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        test_val_split = split_dataset['test'].train_test_split(test_size=0.5)

        train_dataset = split_dataset['train']
        test_dataset = test_val_split['train']
        val_dataset = test_val_split['test']

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, val_loader
