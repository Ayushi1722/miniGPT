import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture"""
    model_type: str = 'gpt-mini'
    vocab_size: int = 50257  # GPT-2 vocabulary size
    block_size: int = 128    # Context size
    n_layer: int = 6         # Number of transformer layers
    n_head: int = 6          # Number of attention heads
    n_embd: int = 192        # Embedding dimension
    embd_pdrop: float = 0.1  # Embedding dropout
    resid_pdrop: float = 0.1 # Residual dropout
    attn_pdrop: float = 0.1  # Attention dropout

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 1
    grad_norm_clip: float = 1.0
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 1000   # Save checkpoint every N steps
    log_every: int = 100     # Log metrics every N steps

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 50
    temperature: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True

@dataclass
class MiniGPTConfig:
    """Main configuration class for MiniGPT"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'MiniGPTConfig':
        """Load configuration from a YAML file"""
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Config file not found: {yaml_file}")
        
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create base config
        config = cls()
        
        # Update model config if present
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        # Update training config if present
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        # Update generation config if present
        if 'generation' in config_dict:
            for k, v in config_dict['generation'].items():
                if hasattr(config.generation, k):
                    setattr(config.generation, k, v)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        return {
            'model': {k: v for k, v in self.model.__dict__.items()},
            'training': {k: v for k, v in self.training.__dict__.items()},
            'generation': {k: v for k, v in self.generation.__dict__.items()},
        }
    
    def save(self, yaml_file: str) -> None:
        """Save configuration to a YAML file"""
        os.makedirs(os.path.dirname(os.path.abspath(yaml_file)), exist_ok=True)
        with open(yaml_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

def get_default_config() -> MiniGPTConfig:
    """Return default configuration for MiniGPT"""
    return MiniGPTConfig() 