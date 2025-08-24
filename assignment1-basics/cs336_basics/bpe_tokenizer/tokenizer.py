import json
import os

from typing import Iterable, Iterator

from cs336_basics.bpe_tokenizer.common import gpt2_bytes_to_unicode
from cs336_basics.bpe_tokenizer.pretokenization import pretokenize_iter, pretokenize_text


class BPETokenizer:

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merge_rules = {(token1, token2): i for i, (token1, token2) in enumerate(merges)}

        self.cache = {}

    def _apply_bpe(self, token: bytes) -> list[int]:
        """
        Apply BPE merges to a token.
        """

        if len(token) <= 1:
            return [self.byte_to_token_id[token]]

        if token in self.cache:
            return self.cache[token]

        token_bytes = [bytes([b]) for b in token]

        while True:
            # all possible merge candidates associated with their ranks
            candidates = [
                (self.merge_rules[pair], i)
                for i in range(len(token_bytes) - 1)
                if (pair := (token_bytes[i], token_bytes[i + 1])) in self.merge_rules
            ]

            if not candidates:
                break
            # find the best merge candidates
            best_rank = min(rank for rank, _ in candidates)
            merge_indexes = {i for rank, i in candidates if rank == best_rank}

            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                if i in merge_indexes:  # apply the merge
                    new_token_bytes.append(token_bytes[i] + token_bytes[i + 1])
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1

            token_bytes = new_token_bytes

        token_id = []
        for byte in token_bytes:
            token_id.append(self.byte_to_token_id[byte])

        self.cache[token] = token_id
        return token_id

    def encode(self, text: str) -> list[int]:
        """
        Encode the input text into a list of token IDs.
        """
        token_ids = []
        tokens = pretokenize_text(text, self.special_tokens)
        for token in tokens:
            if token_id := self.byte_to_token_id.get(token):
                token_ids.append(token_id)
            else:
                token_ids.extend(self._apply_bpe(token))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into a stream of token IDs.
        """
        for text in iterable:
            for token in pretokenize_iter(text, self.special_tokens):
                if token_id := self.byte_to_token_id.get(token):
                    yield token_id
                else:
                    yield from self._apply_bpe(token)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        tokens = [self.vocab[id] for id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        """
        Constructs and returns a Tokenizer from a serialized vocabulary and list of merges(in the same format that your
        BPE training code output) and (optionally) a list of special tokens.

        This is taken from assignment1-basics/tests/test_tokenizer.py:get_tokenizer_from_vocab_merges_path()
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
