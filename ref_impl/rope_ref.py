import torch
import math
import torch.nn as nn

class RoPEEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super(RoPEEmbeddings, self).__init__()
        assert dim % 2 == 0, "dim must be even for RoPE."
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

    # Add the function to rotate half
    def rotate_half(self, x): 
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def build_rope_cache(self):
        # Use this to precompute the RoPE cache
        ...

    def forward(self, pos, x):
        theta = pos.unsqueeze(-1) * self.inv_freq
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return x * cos + self.rotate_half(x) * sin


# Example usage:
dim = 64
batch_size = 2
seq_len = 5

# Create a position tensor for the sequence.
# Here we assume positions 0, 1, ..., seq_len-1.
pos = torch.arange(seq_len)

with torch.no_grad():
    rope = RoPEEmbeddings(dim, max_seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, dim)
    out = rope(pos, x)

    # I thought about verifying this against torchtune RotaryPositionalEmbedding
    # but the rotate_half is different over there. My implementation follows
    # https://github.com/huggingface/transformers/blob/0ed3ffcb4461a244b87781a24e5ebd0a78f98142/src/transformers/models/llama/modeling_llama.py#L159