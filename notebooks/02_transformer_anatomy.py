"""Lab 2: Transformer anatomy experiments and diagnostics."""

import math
import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


def attention_with_weights(q, k, v, mask=None):
    """Diagnostic version: returns both context and attention weights."""
    scores = q @ k.transpose(-2, -1)

    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = torch.softmax(scores, dim=-1)
    context = weights @ v

    return context, weights


def main():
    torch.manual_seed(42)

    # ---------------------------------
    # 1. Numerical equivalence
    # ---------------------------------
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    our_output = attention(q, k, v)
    pytorch_output = F.scaled_dot_product_attention(q, k, v)

    print("Numerical equivalence:")
    print(torch.allclose(our_output, pytorch_output, atol=1e-6))

    # ---------------------------------
    # 2. Multi-Head Attention
    # ---------------------------------
    embed_dim = 64
    num_heads = 4

    mha = MultiHeadAttention(embed_dim, num_heads)

    x = torch.randn(2, 5, embed_dim)
    output = mha(x)

    print("\nMulti-Head Attention:")
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    total_params = sum(
        p.numel() for p in mha.parameters()
    )

    print("\nParameter accounting:")
    print("Total parameters:", total_params)

    # ---------------------------------
    # 3. Causal attention diagnostics
    # ---------------------------------
    seq_len = 4

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len)
    )

    causal_context, causal_weights = attention_with_weights(
        q, k, v, mask=causal_mask
    )

    print("\nCausal mask:")
    print(causal_mask)

    print("\nCausal attention weight matrix:")
    print(causal_weights[0, 0])

    upper_triangle_mass = torch.triu(
        causal_weights[0, 0],
        diagonal=1
    ).sum().item()

    print(
        "Future-attention mass:",
        upper_triangle_mass
    )

    print(
        "Lower-triangular verified:",
        upper_triangle_mass < 1e-6
    )

    print(
        "Model family: Decoder-style causal attention"
    )

    # ---------------------------------
    # 4. Pad-attention leakage
    # ---------------------------------
    seq_len = 5

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    # Example:
    # token, token, [SEP], [PAD], [PAD]
    pad_mask = torch.tensor(
        [[1, 1, 1, 0, 0]]
    )[:, None, None, :]

    _, weights_without_mask = attention_with_weights(
        q, k, v
    )

    _, weights_with_mask = attention_with_weights(
        q, k, v, mask=pad_mask
    )

    pad_mass_without_mask = (
        weights_without_mask[..., 3:]
        .sum(dim=-1)
        .mean()
        .item()
    )

    pad_mass_with_mask = (
        weights_with_mask[..., 3:]
        .sum(dim=-1)
        .mean()
        .item()
    )

    sep_mass_without_mask = (
        weights_without_mask[..., 2]
        .mean()
        .item()
    )

    print("\nAttention-map diagnostics:")
    print(
        "Example layout: token token [SEP] [PAD] [PAD]"
    )

    print(
        "Average [SEP] attention:",
        sep_mass_without_mask
    )

    print(
        "Pad mass WITHOUT mask:",
        pad_mass_without_mask
    )

    print(
        "Pad mass WITH mask:",
        pad_mass_with_mask
    )

    print(
        "Pad leakage fixed:",
        pad_mass_with_mask < 1e-6
    )

    print(
        "\nExample attention matrix without pad mask:"
    )
    print(weights_without_mask[0, 0])

    print(
        "\nExample attention matrix with pad mask:"
    )
    print(weights_with_mask[0, 0])


if __name__ == "__main__":
    main()