from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Config:
    device: str = "cpu"  # Device to use for training, can be "cpu" or "cuda"
    vocab_size: int = 1000  # Vocabulary size, adjusted based on tokenizer
    max_steps: int = 5000  # Maximum number of training steps
    learning_rate: float = 0.001  # Learning rate for the optimizer
    batch_size: int = 5  # Batch size for training
    context_len: int = 256  # Context length for input tokens
    d_model: int = 128  # Embedding dimension
    num_heads: int = 8  # Number of attention heads
    head_dim: int = 0  # Dimension of each attention head
    num_decoders: int = 2  # Number of transformer decoder blcoks
    dropout: float = 0.1  # Dropout rate
    save_path: str = "save/model-ckpt.pth"  # Path to save the model checkpoint
    tokenizer_name: str = (
        "cl100k_base"  # Tokenizer name, can be "cl100k_base" or others
    )

    def __post_init__(self):
        self.head_dim = self.d_model // self.num_heads
        assert (
            self.d_model % self.num_heads == 0
        ), f"d_model({self.d_model}) must be divisible by num_heads({self.num_heads})"
        assert self.context_len > 0, "context_len must be greater than 0"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.vocab_size > 0, "vocab_size must be greater than 0"
        assert self.head_dim > 0, "head_dim must be greater than 0"
        assert self.head_dim % 2 == 0, "head_dim must be even"
        assert self.tokenizer_name in [
            "cl100k_base",
            "gpt2",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "r50k_edit",
            "SimpleBPE",
        ], f"Unsupported tokenizer: {self.tokenizer_name}"
        # make sure the save path directory exists
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return Config(**data)

    @classmethod
    def from_latest_json(cls, directory="."):
        # Find the latest JSON config file in the specified directory
        json_files = sorted(
            (
                f
                for f in Path(directory).iterdir()
                if f.is_file() and f.suffix.lower() == ".json"
            ),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        json_file = str(json_files[0]) if json_files else None

        if json_file is None:
            raise FileNotFoundError("No JSON config file found in the directory.")
        return cls.from_json(json_file), json_file

    def to_json(self, json_file=None):
        if json_file is None:
            json_file = self.save_path + ".json"
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w") as f:
            json.dump(asdict(self), f, indent=4)
        return json_file


# Example usage
if __name__ == "__main__":
    config = Config(save_path="save/testing1.pth")
    print(config)
    saved = config.to_json(config.save_path + ".json")
    print(f"Config saved to {saved}")

    loaded_config, config_file = Config.from_latest_json(f"{Path(saved).parent}")
    print(f"Search latest json under '{Path(saved).parent}/'")
    print(f"Found latest {config_file}")
    print(f"{loaded_config}")
    assert config == loaded_config, "Loaded config does not match original config"
    print(f"(config == loaded_config) is {config == loaded_config}.")
