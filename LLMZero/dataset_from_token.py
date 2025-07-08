import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, tokens, context_len, device=None):
        self.tokens = torch.tensor(tokens, dtype=torch.int32, device=device)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_len]
        y = self.tokens[idx + 1 : idx + self.context_len + 1]
        return x, y


if __name__ == "__main__":
    from tokeniz import Tokenizer
    from config import Config
    from torch.utils.data import DataLoader

    tokenizer = Tokenizer("cl100k_base")
    config = Config()
    print(f"Config: {config}")

    with open("data/dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print(f"Read text from dataset, length: {len(text)}")

    tokens = tokenizer.encode(text)
    dataset = TokenDataset(
        tokens=tokens,
        context_len=config.context_len,
        device=config.device,
    )
    print(f"Dataset length: {len(dataset)}")

    # load a batch of data
    for x, y in DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    ):
        print("A random sample batch:")
        print(f"  x shape: {x.shape}, y shape: {y.shape}")
        print(f"  x: {x}")
        print(f"  y: {y}")
        for i in range(config.batch_size):
            assert torch.all(x[i][1:-1] == y[i][0:-2])
        break
