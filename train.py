import torch
from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer, TokenDataset
from torch.utils.data import DataLoader


def get_dataset(tokenizer, config):
    # Read text as a dataset from a file
    with open("data/dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()
        dataset = TokenDataset(
            tokens=tokenizer.encode(text),
            context_len=config.context_len,
            device=config.device,
        )
        print(
            f"Read text from dataset, length: {len(text)}, number of tokens: {len(dataset)}"
        )
    return dataset


if __name__ == "__main__":
    # Initialize tokenizer and configuration
    tokenizer = Tokenizer("cl100k_base")
    config = Config(
        vocab_size=tokenizer.vocab_size(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"{config}")

    # Create dataset and put it in a DataLoader
    dataset = get_dataset(tokenizer, config)
    dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = MiniGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    # Training loop
    for step, batch in enumerate(dataset, start=1):
        x, y = batch
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Training Loss: {loss.item()}")
