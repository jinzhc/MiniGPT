import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Self attention for multiple heads. See Figure 2 in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, dim, head_size):
        super().__init__()
        self.dim = dim
        self.head_size = head_size

        self.Wq = nn.Linear(dim, head_size)
        self.Wk = nn.Linear(dim, head_size)
        self.Wv = nn.Linear(dim, head_size)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = (q @ k.mT) * (1.0 / (self.head_size**0.5))
        scores = scores.softmax(dim=-1) @ v
        return scores


class MHSA(nn.Module):
    """
    Multi Head Self Attention. See 3.2.2 in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}"
        self.head_size = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [SelfAttention(dim, self.head_size) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch_size, cotext_length, dim)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out


if __name__ == "__main__":
    # Example usage
    batch_size = 3
    context_length = 5
    dim = 2**9
    num_heads = 2
    head_size = dim // num_heads

    x = torch.randn(batch_size, context_length, dim)

    # for attention
    attention1 = SelfAttention(dim, head_size)
    attention2 = SelfAttention(dim, head_size)
    out1 = attention1(x)
    out2 = attention2(x)
    print(f"Input shape: {x.shape}, Output shape: {out1.shape}")
    out = torch.cat([out1, out2], dim=-1)
    print(f"Concatenated output shape: {out.shape}")

    # for multi-head attention
    multihead = MHSA(dim, num_heads)
    out_multihead = multihead(x)
    print(f"Multi-head output shape: {out_multihead.shape}")
    assert out.shape == out_multihead.shape, "Multi-head output shape mismatch"
