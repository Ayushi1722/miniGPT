# MiniGPT

A minimal implementation of the GPT (Generative Pre-trained Transformer) model, motivated by Andrej Karpathy's minGPT implementation.

## Overview

This project implements a smaller version of OpenAI's GPT model. It includes:

- A GPT model implemented from scratch
- A tokenizer based on GPT-2's byte-pair encoding
- Training and text generation functionality
- Shakespeare text dataset for training

The original code was written in a Colab notebook (included as `MiniGPT_CODE.ipynb`) due to the lack of GPU resources on the local machine. A GPU is required if you want to run this code on your local machine; otherwise, the Colab notebook is provided for reference to see the results achieved during inference.

## Credits

This implementation is motivated by Andrej Karpathy's minGPT:
- [@https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)

## Project Structure

```
minigpt/
├── data/
│   ├── datasets.py    # Data loading and preprocessing
│   └── tokenizer.py   # BPE tokenizer
├── model/
│   ├── config.py      # Model configuration
│   └── gpt.py         # GPT model implementation
├── training/
│   └── trainer.py     # Training and text generation utilities
└── main.py            # CLI for training and generating text
```

## Installation

```bash
pip install -r requirements.txt
```

## GPU Requirements

This code requires a GPU for efficient training. If you don't have a GPU available locally:
- Use the provided Colab notebook `MiniGPT_CODE.ipynb`
- Or use a cloud-based GPU service

## Usage

### Training the model

```bash
python -m minigpt.main --train --epochs 1
```

### Generating text

```bash
python -m minigpt.main --generate --prompt "Once upon a time" --max_tokens 100
```

### Training and then generating

```bash
python -m minigpt.main --train --generate --prompt "Once upon a time"
```

## License

This project is open-source and available under the MIT License.
