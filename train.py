from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer
import torch

if __name__ == "__main__":
    # 构建一个分词器
    tokenizer = Tokenizer("cl100k_base")

    # 初始化配置
    config = Config(
        vocab_size=tokenizer.vocab_size(),
        batch_size=5,
        context_len=16,
        num_heads=4,
        num_decoders=3,
        d_model=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"{config}")

    # 加载数据集。为简单起见，我们读取一个文本文件作为数据集。
    with open("data/dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print(f"Read text from dataset, length: {len(text)}")

    # 将文本转换为tokens
    tokens = torch.tensor(
        tokenizer.encode(text), dtype=torch.int32, device=config.device
    )

    # 从tokens中随机获取一个batch，形状为[batch_size, context_len]
    random_idx = torch.randint(
        low=0, high=len(tokens) - config.context_len - 1, size=(config.batch_size,)
    )
    print(f"Indice in X: {random_idx}")
    x = torch.stack([tokens[i : i + config.context_len] for i in random_idx])
    random_idx += 1  # 因为是预测下一个token，所以将index加1
    print(f"Indice in Y: {random_idx}")
    y = torch.stack([tokens[i : i + config.context_len] for i in random_idx])

    model = MiniGPT(config).to(config.device)
    # print(f"Model: {model}")
    out = model(x)
