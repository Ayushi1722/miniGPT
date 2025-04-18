class CfgNode(dict):
    """Simple config dictionary allowing dot notation."""
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def merge_from_dict(self, d): self.update(d)

def get_default_config():
    """Return default configuration for MiniGPT"""
    config = CfgNode()
    config.model_type = 'gpt-mini'
    config.vocab_size = 50257  # GPT-2 vocabulary size
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 192
    config.embd_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.attn_pdrop = 0.1
    config.batch_size = 32
    config.learning_rate = 3e-4
    config.epochs = 1
    config.grad_norm_clip = 1.0
    config.device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    return config
