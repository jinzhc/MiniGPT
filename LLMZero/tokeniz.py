import tiktoken
import gzip
import numpy as np
from pathlib import Path
from collections import Counter


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

    @property
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


class SimpleBPE(object):
    """A minimal BPE tokenizer that can be trained and saved/loaded."""

    def __init__(self):
        super().__init__()
        self.name = "SimpleBPE"
        self._vocab = {}  # mapping from token to ID, {token_id -> bytes}
        self._merges = {}  # mapping from pair to new token, {(old_id,old_id) -> new_id}
    
    def _merge_pair(self, tokens, pair, new_id):
        updated_tokens = []
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                updated_tokens.append(new_id)
                i += 2
            else:
                updated_tokens.append(tokens[i])
                i += 1

        # if the last token was not part of the pair, append it
        if i == len(tokens) - 1:
            updated_tokens.append(tokens[i])

        return updated_tokens

    def train(self, text, vocab_size: int = 10000):
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        tokens = list(text.encode("utf-8"))
        merges = {}  # (token_id,token_id) -> new_token_int
        # init vocab with single byte tokens, total 256 tokens
        vocab = {i: bytes([i]) for i in range(256)}  # token_id -> bytes

        # merge loop
        for new_token_id in range(256, vocab_size):
            # Generate pairs of tokens
            pairs = [(i, j) for i, j in zip(tokens[:-1], tokens[1:])]
            pair, count = Counter(pairs).most_common(1)[0]
            if count <= 10:
                break

            merges[pair] = new_token_id
            vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

            # update tokens with the merged pair
            tokens = self._merge_pair(tokens, pair, new_token_id)

            # logging
            print(
                f"Pair:{pair},count:{count},new_token={new_token_id},bytes={vocab[new_token_id]},length-after-merge={len(tokens)}"
            )

        self._merges = merges
        self._vocab = vocab

    def encode(self, text):
        if len(self._merges) == 0 or len(self._vocab) == 0:
            raise ValueError("Tokenizer must load or be trained before encoding.")
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            # find the pair with the lowest merge index
            pairs_counter = Counter([(i, j) for i, j in zip(tokens[:-1], tokens[1:])])
            pair = min(pairs_counter, key=lambda p: self._merges.get(p, float("inf")))
            if pair not in self._merges:
                break  # nothing else can be merged anymore
            idx = self._merges[pair]
            tokens = self._merge_pair(tokens, pair, idx)
        return tokens

    def decode(self, tokens):
        """Decode token IDs to text."""
        if len(self._merges) == 0 or len(self._vocab) == 0:
            raise ValueError("Tokenizer must load or be trained before encoding.")
        text_bytes = b"".join(self._vocab[idx] for idx in tokens)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save(self, prefix):
        if len(self._merges) == 0 or len(self._vocab) == 0:
            raise ValueError("Tokenizer must load or be trained before encoding.")
        # save _merges and _vocab to npy files
        merges_path = Path(f"save/{prefix}.model.npy")
        vocab_path = Path(f"save/{prefix}.vocab.npy")
        merges_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(merges_path, self._merges)
        np.save(vocab_path, self._vocab)
        return prefix

    @classmethod
    def load(cls, prefix):
        merges_path = Path(f"save/{prefix}.model.npy")
        vocab_path = Path(f"save/{prefix}.vocab.npy")
        tokenizer = cls()
        tokenizer._merges = np.load(merges_path, allow_pickle=True).item()
        tokenizer._vocab = np.load(vocab_path, allow_pickle=True).item()
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._vocab)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    with gzip.open("data/dataset.txt.gz", "rt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = SimpleBPE()
    tokenizer.train(text)
    tokenizer.save("simple_bpe")

    tokenizer_loaded = SimpleBPE.load("simple_bpe")

    text = """In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention.
For translation tasks, the Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014
English-to-French translation tasks, we achieve a new state of the art. In the former task our best
model outperforms even all previously reported ensembles.
We are excited about the future of attention-based models and plan to apply them to other tasks. We
plan to extend the Transformer to problems involving input and output modalities other than text and
to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs
such as images, audio and video. Making generation less sequential is another research goals of ours."""
    print(f"Original text: {text}")
    encoded = tokenizer_loaded.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer_loaded.decode(encoded)
    print(f"Decoded: {decoded}")
    assert text == decoded, "Decoded text does not match original"
