"""Lab 2: Transformer anatomy experiments."""

import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


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

    # ---------------------------------
    # 3. Parameter accounting
    # ---------------------------------
    total_params = sum(
        p.numel() for p in mha.parameters()
    )

    print("\nParameter accounting:")
    print("Total parameters:", total_params)

    # ---------------------------------
    # 4. Causal masking
    # ---------------------------------
    seq_len = 4

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len)
    )

    print("\nCausal mask:")
    print(causal_mask)

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    causal_output = attention(
        q, k, v, mask=causal_mask
    )

    print("Causal attention output shape:")
    print(causal_output.shape)

    # ---------------------------------
    # 5. Pad-attention leakage
    # ---------------------------------
    seq_len = 5

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    # Last two positions are padding
    pad_mask = torch.tensor([
        [1, 1, 1, 0, 0]
    ])

    # Reshape mask for attention scores
    pad_mask = pad_mask[:, None, None, :]

    # Attention without masking padding
    output_without_mask = attention(q, k, v)

    # Attention with padding masked
    output_with_mask = attention(
        q, k, v, mask=pad_mask
    )

    print("\nPad-attention leakage:")
    print("Pad mask:", pad_mask)
    print(
        "Outputs differ after masking:",
        not torch.allclose(
            output_without_mask,
            output_with_mask
        )
    )


if __name__ == "__main__":
    main()