"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)

    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = torch.softmax(scores, dim=-1)

    context = weights @ v

    return context


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        context = attention(q, k, v, mask)

        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            seq_len,
            self.embed_dim,
        )

        output = self.out_proj(context)

        return output