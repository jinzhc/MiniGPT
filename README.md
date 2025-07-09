# Text2Generate

## Introduction

Build a Mini GPT from zero, only based on PyTorch.

The primary objective is to acquire knowledge regarding Large Language Models (LLMs).

## Current Status

Currently it also requires a tokenizer from tiktoken. Next will also build a BPE from scratch.

## Dependency

Clone this project.

```Bash
git clone <this_project>
cd <project_folder>
```

Setup the environment using [uv](https://docs.astral.sh/uv/getting-started/installation/).

```Bash
uv venv --python 3.13
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy
uv pip install tiktoken
```

## Training

```Bash
uv run train.py
```

Generally, it will take 4 to 5 hours.

## Generate

After done the training. This script will search the latest saved model and using it to generate.

```Bash
uv run generate.py
```

## Do it yourself

1. Create your own config in `./train.py`. Can see `Config` dataclass in `LLMZero/config.py`.

    ```Python
    tokenizer_name = "cl100k_base"  # from tiktoken
    tokenizer = Tokenizer(tokenizer_name)
    config = Config(
        save_path="save/mini_gpt001.pth",
        tokenizer_name=tokenizer_name,
        vocab_size=tokenizer.vocab_size(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"{config}")
    ```

2. Optional, update `data/dataset.txt` with your data.
3. Train it and generate with your model.

## License

MIT License

Copyright (c) 2025 Jin Zhengao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
