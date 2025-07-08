import torch
from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer, TokenDataset
from torch.utils.data import DataLoader


def get_dataset(tokenizer, config):
    # 加载数据集并转为tokens。为简单起见，我们读取一个文本文件作为数据集。
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
    # 初始化配置
    tokenizer = Tokenizer("cl100k_base")
    config = Config(
        vocab_size=tokenizer.vocab_size(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"{config}")

    dataset = get_dataset(tokenizer, config)
    dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = MiniGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
    # print(f"Model: {model}")

    for step, batch in enumerate(dataset, start=1):
        x, y = batch
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Training Loss: {loss.item()}")
