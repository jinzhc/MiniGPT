import torch
import math
from .attention import MHSA
from .config import Config


def positional_encoding(d_model, length):
    """
    Sinusoidal positional encoding. See section 3.5 in "Attention Is All You Need".
    """
    assert d_model % 2 == 0, f"d_model {d_model} must be even for positional encoding"
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class FFN(torch.nn.Module):
    """
    Feed Forward Network (FFN) in Transformer decoder block.

    See section 3.3 in "Attention Is All You Need".
    """

    def __init__(self, config: Config):
        super().__init__()
        self.d_ff = 4 * config.d_model
        self.linear1 = torch.nn.Linear(config.d_model, self.d_ff)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.d_ff, config.d_model)
        # self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.dropout(x)
        return x


class TransformerDecoder(torch.nn.Module):
    """
    The decoder in transformer architecture.

    See Figure 1 in GPT1 tech report "Improving Language Understandingby Generative Pre-Training" by OpenAI.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.attn = MHSA(config)
        self.ln1 = torch.nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)
        self.ln2 = torch.nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the compute order in decoder: mha, residual add, layer norm1, feed-forward, residual add, layer norm2
        x = x + self.attn(x)  # Multi-head attention
        x = self.ln1(x)  # Residual connection and layer normalization
        x = x + self.ffn(x)  # Feed-forward network
        x = self.ln2(x)  # Residual connection and layer normalization
        return x


class MiniGPT(torch.nn.Module):
    """
    MiniGPT model, a simplified version of GPT architecture.
    """

    def __init__(self, config: Config):
        super().__init__()

        # construct the embedding layer
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)

        # construct the transformer decoder blocks
        self.blocks = torch.nn.Sequential(
            *(TransformerDecoder(config) for _ in range(config.num_decoders))
        )

        # construct the final linear layer, prepare to predict the next token
        self.final_layer = torch.nn.Linear(config.d_model, config.vocab_size)
        self.positional_embedding = positional_encoding(
            config.d_model, config.context_len
        ).to(device=config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch_size, context_length)
        print(f"MiniGPT.forward: x.shape = {x.shape}")

        x = self.embedding(x)  # Convert token to embeddings
        x += self.positional_embedding  # add positional embedding
        x = self.blocks(x)  # Pass through transformer blocks
        logits = self.final_layer(x)  # Final linear layer

        print(f"MiniGPT.forward: logits.shape = {logits.shape}")
        return logits
