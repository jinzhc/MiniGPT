import torch
import math

try:
    from .config import Config
    from .attention import MHSA
except ImportError:
    # If running as a script, use absolute import
    from config import Config
    from attention import MHSA


def positional_encoding(d_model, length):
    """
    Sinusoidal positional encoding. See section 3.5 in "Attention Is All You Need".
    """
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
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
        self.final_projection_layer = torch.nn.Linear(config.d_model, config.vocab_size)

        # positional encoding
        self.register_buffer(
            "positional_embedding",
            positional_encoding(config.d_model, config.context_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch_size, context_len)
        x = self.embedding(x)  # Convert token to embeddings
        x += self.positional_embedding  # add positional embedding
        x = self.blocks(x)  # Pass through transformer blocks
        logits = self.final_projection_layer(x)  # Final linear layer
        return logits


if __name__ == "__main__":
    # Example usage
    config = Config()
    model = MiniGPT(config).to(config.device)
    print(model)

    # Create a dummy input tensor
    idx = torch.randint(
        low=0, high=config.vocab_size, size=(config.batch_size, config.context_len)
    ).to(config.device)
    out = model(idx)
    print(config)
    print(f"Input and Output shape: {idx.shape}, {out.shape}")
