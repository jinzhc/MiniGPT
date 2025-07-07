from LLMZero import MiniGPT
from LLMZero import Tokenizer
from dataclasses import dataclass

@dataclass
class ModelConfig:
    embed_dim: int = 64 # Embedding dimension
    num_heads: int = 4 # Number of attention heads
    block_size: int = 2 # number of decoder blocks


if __name__ == "__main__":
    tokenizer = Tokenizer()
    input_text = "为人民服务！"
    tokens = tokenizer.encode(input_text)
    decoded_text = tokenizer.decode(tokens)
    print(f"tokenizer.encode('{input_text}') = {tokens}")
    print(f"tokenizer.decode('{tokens}') = '{decoded_text}'")

    model = MiniGPT().
    print(f"Model: {model}")

    pass
