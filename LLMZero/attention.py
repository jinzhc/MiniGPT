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
        # x: (batch_size, context_len, d_model)
        batch_size, context_len, _ = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = (q @ k.mT) * (1.0 / (self.head_dim**0.5))
        mask = torch.ones(context_len, context_len, device=scores.device)
        mask = torch.triu(mask, diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
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
        # x: (batch_size, context_len, d_model)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out


if __name__ == "__main__":
    # Example usage
    config = Config(
        num_heads=2,
    )

    # for attention
    x = torch.randn((config.batch_size, config.context_len, config.d_model))
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
