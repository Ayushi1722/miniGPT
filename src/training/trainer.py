import os
import time
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ..data.tokenizer import Encoder
from ..utils.logger import logger, training_logger
from ..model.gpt import GPT

class Trainer:
    """
    Trainer class for training GPT models.
    
    This handles the training loop, evaluation, checkpointing, and inference.
    """
    def __init__(
        self,
        model: GPT,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = None,
        val_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GPT model to train
            train_dataloader: DataLoader for training data
            optimizer: Optimizer to use
            config: Training configuration
            device: Device to train on ('cuda' or 'cpu')
            val_dataloader: Optional DataLoader for validation data
            lr_scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.lr_scheduler = lr_scheduler
        
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.log_every = config.get('log_every', 100)
        self.save_every = config.get('save_every', 1000)
        self.grad_norm_clip = config.get('grad_norm_clip', 1.0)
        
        logger.info(f"Initialized trainer with model on device: {self.device}")
    
    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """
        Save a training checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a training checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint {filepath} does not exist, skipping loading")
            return
        
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        training_logger.start_epoch(self.epoch)
        
        for step, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            training_logger.start_step()
            
            logits, loss = self.model(x, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            total_loss += loss.item()
            
            self.global_step += 1
            
            if step % self.log_every == 0:
                training_logger.log_step(
                    step=step,
                    loss=loss.item(),
                    lr=current_lr,
                    epoch=self.epoch
                )
            
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"model_step_{self.global_step}.pt")
        
        avg_loss = total_loss / num_batches
        
        training_logger.end_epoch(self.epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            logger.warning("No validation dataloader provided, skipping evaluation")
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        logger.info("Evaluating on validation set...")
        
        for x, y in self.val_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            logits, loss = self.model(x, y)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self, num_epochs: int) -> GPT:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            Trained model
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        training_logger.start_training()
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch} completed with average loss: {train_loss:.4f}")
            
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(f"model_epoch_{epoch}.pt", is_best=is_best)
            else:
                self.save_checkpoint(f"model_epoch_{epoch}.pt")
        
        training_logger.end_training()
        logger.info("Training completed")
        
        return self.model

def generate_text(
    model: GPT, 
    tokenizer: Encoder, 
    prompt: str, 
    max_new_tokens: int = 50, 
    temperature: float = 0.9, 
    do_sample: bool = True, 
    top_k: Optional[int] = None,
    device: str = None
) -> str:
    """
    Generate text using a trained model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling (higher = more random)
        do_sample: Whether to sample or take the most likely token
        top_k: If set, only consider the top k tokens
        device: Device to run inference on
        
    Returns:
        Generated text string
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()
    
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    logger.info(f"Generating text with prompt: '{prompt}'")
    with torch.no_grad():
        output = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature,
            top_k=top_k
        )
        generated = tokenizer.decode(output[0].tolist())
    
    logger.info(f"Generated text of {len(generated)} characters")
    
    return generated 