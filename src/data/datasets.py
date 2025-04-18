import os
import requests
from typing import Union, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.logger import logger
from .tokenizer import Encoder

@dataclass
class DataConfig:
    """Configuration for datasets"""
    dataset_name: str = "shakespeare"
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "minigpt")
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

class CharDataset(Dataset):
    """
    Character-level dataset for training language models.
    
    This dataset takes a text input and creates training examples based on
    a sliding window approach. Each example is a sequence of tokens of length
    block_size where the target is the same sequence but shifted by one position.
    """
    def __init__(self, text: str, tokenizer: Encoder, block_size: int):
        """
        Initialize the dataset.
        
        Args:
            text: Raw text to tokenize and create examples from
            tokenizer: Tokenizer instance to convert text to token ids
            block_size: Size of context window for examples
        """
        logger.info(f"Creating CharDataset with block size {block_size}")
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Encode the full text
        logger.info("Encoding text...")
        self.data = tokenizer.encode(text)
        logger.info(f"Encoded {len(text)} characters into {len(self.data)} tokens")

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        # We can create len(data) - block_size examples
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Tuple of (input, target) tensors
        """
        # Extract a chunk of block_size + 1 tokens (we need the +1 for the target)
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Split into input (x) and target (y)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def load_shakespeare_dataset(tokenizer: Encoder, 
                            config: DataConfig,
                            block_size: int) -> CharDataset:
    """
    Download and load Shakespeare dataset.
    
    Args:
        tokenizer: Tokenizer instance to use
        config: Data configuration
        block_size: Size of context window for examples
        
    Returns:
        CharDataset instance
    """
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = os.path.join(config.cache_dir, "shakespeare.txt")
    
    # Make sure cache directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Download the dataset if needed
    if not os.path.isfile(data_path):
        logger.info(f"Downloading Shakespeare dataset from {data_url}")
        r = requests.get(data_url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(r.text)
        logger.info(f"Downloaded Shakespeare dataset to {data_path}")
    else:
        logger.info(f"Using cached Shakespeare dataset at {data_path}")

    # Read the text file
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Loaded {len(text)} characters of text")
    
    # Create the dataset
    return CharDataset(text, tokenizer, block_size)


def get_dataloader(dataset: Dataset, 
                  config: DataConfig, 
                  is_train: bool = True) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset to create loader for
        config: Data configuration
        is_train: Whether this is for training or validation
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train and config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )


def get_dataset(tokenizer: Encoder, 
               config: DataConfig,
               block_size: int) -> Dataset:
    """
    Get a dataset based on the dataset name in the config.
    
    Args:
        tokenizer: Tokenizer instance
        config: Data configuration
        block_size: Size of context window
        
    Returns:
        Dataset instance
    """
    if config.dataset_name.lower() == "shakespeare":
        return load_shakespeare_dataset(tokenizer, config, block_size)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}") 