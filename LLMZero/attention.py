import torch
from torch import nn

try:
    from .config import Config
except ImportError:
    # If running as a script, use absolute import
    from config import Config

class SelfAttention(nn.Module):
    """
    Self attention for multiple heads. See Figure 2 in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.Wq = nn.Linear(config.d_model, config.head_dim)
        self.Wk = nn.Linear(config.d_model, config.head_dim)
        self.Wv = nn.Linear(config.d_model, config.head_dim)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = (q @ k.mT) * (1.0 / (self.head_dim**0.5))
        scores = scores.softmax(dim=-1) @ v
        return scores


class MHSA(nn.Module):
    """
    Multi Head Self Attention. See 3.2.2 in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(config) for _ in range(config.num_heads)]
        )
        self.projection = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        # x: (batch_size, context_len, dim)
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

    # Create config object
    config = Config(
        d_model=dim,
        num_heads=num_heads,
        head_dim=head_size
    )

    # for attention
    attention1 = SelfAttention(config)
    attention2 = SelfAttention(config)
    out1 = attention1(x)
    out2 = attention2(x)
    print(f"Input shape: {x.shape}, Output shape: {out1.shape}")
    out = torch.cat([out1, out2], dim=-1)
    print(f"Concatenated output shape: {out.shape}")

    # for multi-head attention
    multihead = MHSA(config)
    out_multihead = multihead(x)
    print(f"Multi-head output shape: {out_multihead.shape}")
    assert out.shape == out_multihead.shape, "Multi-head output shape mismatch"
