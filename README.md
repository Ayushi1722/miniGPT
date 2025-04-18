# MiniGPT

A minimal implementation of the GPT (Generative Pre-trained Transformer) model, motivated by Andrej Karpathy's minGPT implementation.

## Overview

This project implements a smaller version of OpenAI's GPT model. It includes:

- A GPT model implemented from scratch
- A tokenizer based on GPT-2's byte-pair encoding
- Training and text generation functionality
- Shakespeare text dataset for training

The original code was written in a Colab notebook (included in `notebooks/MiniGPT_CODE.ipynb`) due to the lack of GPU resources on the local machine. A GPU is required if you want to run this code on your local machine; otherwise, the Colab notebook is provided for reference to see the results achieved during inference.

## Credits

This implementation is motivated by Andrej Karpathy's minGPT:
- [@https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)

## Project Structure

```
.
├── notebooks/                    # Jupyter notebooks
│   └── MiniGPT_CODE.ipynb        # Original implementation
├── src/                          # Source code
│   ├── config/                   # Configuration
│   │   ├── config.py             # Config classes
│   │   └── default_config.yaml   # Default configuration
│   ├── data/                     # Data handling
│   │   ├── datasets.py           # Dataset classes
│   │   └── tokenizer.py          # BPE tokenizer
│   ├── model/                    # Model implementation
│   │   └── gpt.py                # GPT model architecture
│   ├── training/                 # Training utilities
│   │   └── trainer.py            # Trainer class
│   ├── utils/                    # Utilities
│   │   └── logger.py             # Logging utilities
│   ├── __init__.py               # Package initialization
│   ├── __main__.py               # Entry point for module execution
│   └── cli.py                    # Command line interface
├── .gitignore                    # Git ignore file
├── setup.py                      # Package setup script
├── README.md                     # Project documentation
└── requirements.txt              # Package dependencies
```

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/minigpt.git
cd minigpt

# Install the package
pip install -e .
```

### Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

## GPU Requirements

This code requires a GPU for efficient training. If you don't have a GPU available locally:
- Use the provided Colab notebook in `notebooks/MiniGPT_CODE.ipynb`
- Or use a cloud-based GPU service

## Usage

### Command Line Interface

The package provides a command-line interface for training and generating text:

```bash
# Show help
minigpt --help

# Show version
minigpt version

# Train a model (with default parameters)
minigpt train

# Train with custom parameters
minigpt train --epochs 3 --batch-size 64 --learning-rate 2e-4

# Generate text
minigpt generate --model checkpoints/model_best.pt --prompt "Once upon a time" --max-tokens 100 --temperature 0.8
```

### Python API

```python
from src.model.gpt import GPT
from src.config.config import get_default_config
from src.data.tokenizer import get_encoder
from src.training.trainer import generate_text
import torch

# Load a model
config = get_default_config()
model = GPT(config.model)
model.load_state_dict(torch.load("checkpoints/model_best.pt")["model_state_dict"])

# Generate text
tokenizer = get_encoder()
text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

## License

This project is open-source and available under the MIT License.
