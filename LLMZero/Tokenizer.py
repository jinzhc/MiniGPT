import tiktoken


class Tokenizer(object):
    def __init__(self, name="cl100k_base"):
        print(f"Tokenizer.__init__")
        self.name = name
        assert name in [
            "cl100k_base",
            "gpt2",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "r50k_edit",
        ], f"Unsupported tokenizer: {name}"
        self.tokenizer = tiktoken.get_encoding(name)

    def encode(self, text):
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text

    def __call__(self, text):
        return self.encode(text)
