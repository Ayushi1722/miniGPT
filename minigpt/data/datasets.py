import os
import requests
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_shakespeare_dataset(tokenizer, config):
    """Download and load Shakespeare dataset"""
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = os.path.join(os.path.expanduser("~"), ".cache", "mingpt", "shakespeare.txt")
    
    if not os.path.isfile(data_path):
        print("Downloading Tiny Shakespeare dataset...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        r = requests.get(data_url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(r.text)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    return CharDataset(text, tokenizer, config.block_size)
