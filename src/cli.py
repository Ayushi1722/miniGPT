#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Dict, Any

import torch

from .model.gpt import GPT
from .config.config import MiniGPTConfig, get_default_config
from .data.tokenizer import get_encoder
from .data.datasets import DataConfig, get_dataset, get_dataloader
from .training.trainer import Trainer, generate_text
from .utils.logger import logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MiniGPT: Train and generate text with a GPT model"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: use built-in config)'
    )
    train_parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    train_parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of epochs to train for (overrides config)'
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for training (overrides config)'
    )
    train_parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Learning rate for training (overrides config)'
    )
    train_parser.add_argument(
        '--device', type=str, default=None,
        help='Device to train on (cuda or cpu, default: use GPU if available)'
    )
    
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    gen_parser.add_argument(
        '--prompt', type=str, default="Once upon a time",
        help='Text prompt to start generation with'
    )
    gen_parser.add_argument(
        '--max-tokens', type=int, default=50,
        help='Maximum number of tokens to generate'
    )
    gen_parser.add_argument(
        '--temperature', type=float, default=0.9,
        help='Temperature for sampling (higher = more random)'
    )
    gen_parser.add_argument(
        '--top-k', type=int, default=None,
        help='If set, only sample from the top k most likely tokens'
    )
    gen_parser.add_argument(
        '--no-sample', action='store_true',
        help='Don\'t sample, just take the most likely token each time'
    )
    gen_parser.add_argument(
        '--device', type=str, default=None,
        help='Device to run on (cuda or cpu, default: use GPU if available)'
    )
    
    subparsers.add_parser('version', help='Show version info')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return args

def train_model(args):
    """Train a model with the given arguments"""
    if args.config is not None:
        config = MiniGPTConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        logger.info(f"Overriding batch size: {args.batch_size}")
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
        logger.info(f"Overriding learning rate: {args.learning_rate}")
    
    if args.device is not None:
        config.training.device = args.device
        logger.info(f"Overriding device: {args.device}")
    
    device = config.training.device
    logger.info(f"Using device: {device}")
    
    tokenizer = get_encoder()
    
    data_config = DataConfig(
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )
    
    dataset = get_dataset(
        tokenizer=tokenizer,
        config=data_config,
        block_size=config.model.block_size
    )
    
    train_dataloader = get_dataloader(
        dataset=dataset,
        config=data_config,
        is_train=True
    )
    
    model = GPT(config.model)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.1
    )
    
    # Linear warmup then cosine decay
    def lr_lambda(step):
        warmup_steps = int(0.1 * config.training.epochs * len(train_dataloader))
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        decay_steps = config.training.epochs * len(train_dataloader) - warmup_steps
        step = step - warmup_steps
        return 0.5 * (1.0 + torch.cos(torch.tensor(step / decay_steps * 3.14159)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        config=vars(config.training),
        device=device,
        lr_scheduler=scheduler
    )
    
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)
    
    trainer.train(config.training.epochs)
    
    return trainer

def generate_from_model(args):
    """Generate text with the given arguments"""
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = get_default_config()
        for k, v in config_dict.items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)
    else:
        config = get_default_config()
    
    model = GPT(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {args.model}")
    
    tokenizer = get_encoder()
    
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
        top_k=args.top_k,
        device=device
    )
    
    print("\n" + "="*50)
    print(f"PROMPT: {args.prompt}")
    print("-"*50)
    print(f"GENERATED:\n{generated}")
    print("="*50)

def show_version():
    """Show version info"""
    print("MiniGPT v0.1.0")
    print("A minimal implementation of GPT inspired by Andrej Karpathy's minGPT")
    print("https://github.com/karpathy/minGPT")

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_from_model(args)
    elif args.command == 'version':
        show_version()
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
