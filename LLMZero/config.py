from dataclasses import dataclass


@dataclass
class Config:
    device: str = "cpu"  # Device to use for training, can be "cpu" or "cuda"
    batch_size: int = 4  # Batch size for training
    context_len: int = 32  # Context length for input tokens
    d_model: int = 64  # Embedding dimension
    vocab_size: int = 1000  # Vocabulary size, can be adjusted based on tokenizer
    num_heads: int = 4  # Number of attention heads
    num_decoders: int = 2  # Number of transformer decoder
    dropout: float = 0.1  # Dropout rate
