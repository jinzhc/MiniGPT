from dataclasses import dataclass


@dataclass
class Config:
    device: str = "cpu"  # Device to use for training, can be "cpu" or "cuda"
    vocab_size: int = 1000  # Vocabulary size, adjusted based on tokenizer
    batch_size: int = 5  # Batch size for training
    context_len: int = 16  # Context length for input tokens
    d_model: int = 64  # Embedding dimension
    num_heads: int = 2  # Number of attention heads
    head_dim: int = 32  # Dimension of each attention head
    num_decoders: int = 3  # Number of transformer decoder blcoks
    dropout: float = 0.1  # Dropout rate

    def __post_init__(self):
        self.head_dim = self.d_model // self.num_heads
        assert (
            self.d_model % self.num_heads == 0
        ), f"d_model({self.d_model}) must be divisible by num_heads({self.num_heads})"
        assert self.context_len > 0, "context_len must be greater than 0"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.vocab_size > 0, "vocab_size must be greater than 0"
        assert self.head_dim > 0, "head_dim must be greater than 0"
