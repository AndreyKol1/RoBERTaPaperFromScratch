import pickle
from collections import defaultdict
from itertools import chain

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPE:
    def __init__(self, regex_pattern=GPT4_SPLIT_PATTERN):
        self.regex_pattern = regex_pattern
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}

    def re_to_bytes(self, text):
        text_splitted = re.findall(self.regex_pattern, text)
        ids = [list(word.encode("utf-8")) for word in text_splitted]
        return list(chain(*ids))

    def get_stats(self, tokens):
        vocab = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            vocab[pair] += 1

        return vocab

    def create_merges(self, pair, idx, ids):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text, vocab_size, verbose=False):
        tokens = self.re_to_bytes(text)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(255, vocab_size):
            stat = self.get_stats(tokens)
            pair = max(stat, key=stat.get)
            tokens = self.create_merges(pair, i, tokens)
            self.merges[pair] = i
            self.vocab[i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"Pair {pair} is merged into {i}")

    def encode(self, text):
        tokens = self.re_to_bytes(text)
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = self.create_merges(pair, idx, tokens)

        return tokens

    def decode(self, ids):
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8")

    def add_special_tokens(self, tokens: list):
        if not hasattr(self, "special_tokens"):
            self.special_tokens = {}

        current_max_id = max(self.vocab.keys()) + 1

        for token in tokens:
            token_bytes = token.encode("utf-8")
            self.vocab[current_max_id] = token_bytes
            self.special_tokens[token] = current_max_id
            current_max_id += 1

    def save(self, filename):
        assert filename.endswith(".pickle"), (
            "file should be saved only with pickle extension."
        )

        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "regex_pattern": self.regex_pattern,
        }

        with open(f"{filename}", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename):
        assert filename.endswith(".pickle"), (
            "file should be loaded only with pickle extension."
        )

        with open(f"{filename}", "rb") as f:
            data = pickle.load(f)

        # creates instance of class with a regex expression as class requires
        tokenizer = cls(data["regex_pattern"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data["merges"]

        return tokenizer
