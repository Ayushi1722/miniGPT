import os
import sys
import logging
import time
from typing import Optional

class Logger:
    """
    Simple logging utility that logs to both console and file
    """
    def __init__(
        self, 
        name: str = "minigpt",
        log_dir: str = "logs",
        log_level: int = logging.INFO
    ):
        self.name = name
        self.log_dir = log_dir
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log_dir is specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, 
                f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.info(f"Logger initialized. Logs will be saved to: {log_dir}")
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)

class TrainingLogger(Logger):
    """
    Logger specifically for training, with additional methods for tracking metrics
    """
    def __init__(
        self, 
        name: str = "training",
        log_dir: str = "logs",
        log_level: int = logging.INFO
    ):
        super().__init__(name, log_dir, log_level)
        self.training_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None
        self.metrics = {}
    
    def start_training(self):
        """Mark the start of training"""
        self.training_start_time = time.time()
        self.info("Training started.")
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
        self.info(f"Epoch {epoch} started.")
    
    def start_step(self):
        """Mark the start of a step"""
        self.step_start_time = time.time()
    
    def log_step(self, step: int, loss: float, lr: float, epoch: Optional[int] = None):
        """Log step information"""
        if self.step_start_time is None:
            self.warning("step_start_time is None. Did you call start_step()?")
            step_time = 0
        else:
            step_time = time.time() - self.step_start_time
        
        # Build epoch string if epoch is provided
        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        
        # Log the metrics
        self.info(
            f"{epoch_str}Step {step}: "
            f"loss={loss:.4f}, "
            f"lr={lr:.6f}, "
            f"step_time={step_time:.3f}s"
        )
        
        # Store metrics
        self.metrics[step] = {
            'loss': loss, 
            'lr': lr, 
            'step_time': step_time
        }
        if epoch is not None:
            self.metrics[step]['epoch'] = epoch
    
    def end_epoch(self, epoch: int, val_loss: Optional[float] = None):
        """Mark the end of an epoch and log metrics"""
        if self.epoch_start_time is None:
            self.warning("epoch_start_time is None. Did you call start_epoch()?")
            epoch_time = 0
        else:
            epoch_time = time.time() - self.epoch_start_time
        
        # Log basic end of epoch info
        log_msg = f"Epoch {epoch} completed in {epoch_time:.2f}s."
        
        # Add validation loss if provided
        if val_loss is not None:
            log_msg += f" Validation loss: {val_loss:.4f}"
        
        self.info(log_msg)
    
    def end_training(self):
        """Mark the end of training and log total time"""
        if self.training_start_time is None:
            self.warning("training_start_time is None. Did you call start_training()?")
            total_time = 0
        else:
            total_time = time.time() - self.training_start_time
        
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.info(
            f"Training completed in "
            f"{int(hours)}h {int(minutes)}m {seconds:.2f}s."
        )

# Global logger instances
logger = Logger()
training_logger = TrainingLogger()
