import torch
from .Attention import MultiHeadAttention

"""
"""
class MiniGPT(torch.nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, block_size=2):
        super().__init__()
        # 初始化
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"MiniGPT.forward: x.shape = {x.shape}")
        return x