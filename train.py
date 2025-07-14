import torch
from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer, TokenDataset, SimpleBPE
from torch.utils.data import DataLoader
from datetime import datetime
import gzip


def get_dataset(tokenizer, config):
    # Read text as a dataset from a file
    with gzip.open(config.corpus, "rt", encoding="utf-8") as f:
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
    # tokenizer_name = "cl100k_base"  # from tiktoken
    # tokenizer = Tokenizer(tokenizer_name)
    
    config = Config(
        save_path="save/mini_gpt001.pth",
        tokenizer_name="SimpleBPE",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=50000,
    )
    with gzip.open(config.corpus, "rt", encoding="utf-8") as f:
        text = f.read()
    tokenizer = SimpleBPE()
    tokenizer.train(text)
    config.vocab_size = tokenizer.vocab_size
    tokenizer.save(config.tokenizer_name)

    print(f"{config}")

    # Create dataset and put it in a DataLoader
    dataset = get_dataset(tokenizer, config)
    dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = MiniGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

    # Training loop
    start_time = datetime.now()
    for step, batch in enumerate(dataset, start=1):
        x, y = batch
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:08d}, Training Loss: {loss.item():.8f}")

        if step >= config.max_steps:
            break

    interval = datetime.now() - start_time
    print(f"Training completed in {interval.total_seconds() / (60*60.0):.2f} hours.")

    # Save the model
    saved_config_file = model.save()
    print(
        f"Model saved to '{model.config.save_path}' and its configuration saved to '{saved_config_file}'."
    )
