from torch.utils.data import Dataset
import pandas as pd
import re
import torch
import os

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', '', text)  
    text = ' '.join(text.split())
    return text

class ChunkedTokenizerSaver:
    def __init__(self, tokenizer, chunk_size=10000, max_len=512):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_and_save(self, raw_data_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        reader = pd.read_csv(raw_data_path, chunksize=self.chunk_size)

        for i, chunk in enumerate(reader):
            print(f"🔹 Processing chunk {i}...")
            texts = [preprocess(t) for t in chunk["text"].tolist()]
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pth"
            )
            tokens["labels"] = tokens["input_ids"].clone()

            chunk_path = os.path.join(save_dir, f"chunk_{i}.pth")
            torch.save(tokens, chunk_path)
            print(f"✅ Saved: {chunk_path}")


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
            data = torch.load(file)
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
    


