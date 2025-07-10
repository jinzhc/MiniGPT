import tiktoken

class Tokenizer(object):
    def __init__(self, name="cl100k_base"):
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

    def vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, text):
        tokens = self.tokenizer.encode(text, allowed_special="all")
        return tokens

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text
    
    @property
    def eot_token(self):
        """End of text token."""
        return self.tokenizer.eot_token

    def __call__(self, text):
        return self.encode(text)
