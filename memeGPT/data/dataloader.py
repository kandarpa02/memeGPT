from torch.utils.data import Dataset
import pandas as pd
import re
import torch
import os
import random

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', '', text)  
    text = ' '.join(text.split())
    return text

class ChunkedTokenizerSaver:
    def __init__(self, tokenizer, chunk_size=10000, max_len=128):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_and_save(
        self, raw_data_path, save_dir,
        split=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    ):
        os.makedirs(save_dir, exist_ok=True)
        reader = pd.read_csv(raw_data_path)

        texts = [preprocess(t) for t in reader["text"].tolist()]
        total = len(texts)

        if split:
            random.seed(seed)
            random.shuffle(texts)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            splits = {
                "train": texts[:train_end],
                "val": texts[train_end:val_end],
                "test": texts[val_end:]
            }
        else:
            splits = {"train": texts}  # Use all data as training data

        for split_name, split_texts in splits.items():
            split_dir = os.path.join(save_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for i in range(0, len(split_texts), self.chunk_size):
                chunk_texts = split_texts[i:i + self.chunk_size]
                print(f"🔹 Processing {split_name} chunk {i // self.chunk_size}...")

                tokens = self.tokenizer(
                    chunk_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                tokens["labels"] = tokens["input_ids"].clone()

                chunk_path = os.path.join(split_dir, f"chunk_{i // self.chunk_size}.pth")
                torch.save(tokens, chunk_path)
                print(f"Saved: {chunk_path}")


class T3nsorLoader(Dataset):
    def __init__(self, chunk_folder):
        self.chunk_files = sorted([
            os.path.join(chunk_folder, f)
            for f in os.listdir(chunk_folder)
            if f.endswith(".pth")
        ])
        self.sample_index = []
        self.chunk_data = None
        self.current_chunk_id = -1

        for chunk_id, file in enumerate(self.chunk_files):
            data = torch.load(file, weights_only=False)
            num_samples = data["input_ids"].shape[0]
            self.sample_index.extend([(chunk_id, i) for i in range(num_samples)])

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        chunk_id, local_idx = self.sample_index[idx]

        if chunk_id != self.current_chunk_id:
            # Load new chunk only if necessary
            self.chunk_data = torch.load(self.chunk_files[chunk_id])
            self.current_chunk_id = chunk_id

        item = {
            key: self.chunk_data[key][local_idx]
            for key in self.chunk_data
        }
        return item
    


