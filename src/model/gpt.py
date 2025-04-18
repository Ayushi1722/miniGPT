import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

class CausalSelfAttention(nn.Module):
    """
    A causal self-attention layer. 
    This is the core mechanism of the transformer architecture, allowing tokens
    to attend to previous tokens in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        # Calculate query, key, value for attention
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention weights
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    """
    Transformer block: communication followed by computation.
    This corresponds to one layer in a transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # Layer norm -> attention -> residual connection
        x = x + self.attn(self.ln_1(x))
        # Layer norm -> MLP -> residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    The full GPT language model, with a context size of block_size.
    This implements a complete GPT-style decoder-only transformer language model.
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled initialization to account for residual connections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """Initialize model weights using best practices from the literature"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model
        Args:
            idx: Tensor of token indices (batch_size, seq_len)
            targets: Optional target tokens for calculating loss
            
        Returns:
            logits: Raw token predictions (batch_size, seq_len, vocab_size)
            loss: Optional calculated loss if targets provided
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Positions for the position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Forward pass through the transformer
        # - Token embeddings + position embeddings
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        # - Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # - Final layer norm
        x = self.transformer.ln_f(x)
        # - Project to vocabulary
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Generate text using the trained model.
        
        Args:
            idx: Context token ids (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Controls randomness in sampling
            do_sample: Whether to sample or take the most likely token
            top_k: If set, only consider the top k tokens
            
        Returns:
            Tensor of token ids including the generated tokens
        """
        for _ in range(max_new_tokens):
            # If the sequence is too long, truncate it to fit the block size
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on the last token
            logits = logits[:, -1, :] / temperature
            # Optional: apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution or take the most likely token
            idx_next = torch.multinomial(probs, num_samples=1) if do_sample else torch.topk(probs, 1, dim=-1)[1]
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx 