import heapq
import json
import os

from collections import Counter, defaultdict
from dataclasses import dataclass

from cs336_basics.bpe_tokenizer.common import gpt2_bytes_to_unicode
from cs336_basics.bpe_tokenizer.pretokenization import pretokenize_file


@dataclass
class PairItem:
    token1: bytes
    token2: bytes
    count: int

    def __lt__(self, other):
        """
        Compare two PairItem instances.
        This is used to maintain the heap property (max-heap).
        """
        if self.count != other.count:
            return self.count > other.count
        if self.token1 != other.token1:
            return self.token1 > other.token1
        return self.token2 > other.token2


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

        self.vocab_set = set()  # keep track of all unique byte values

    def _init_vocab(self):
        """
        Initialize the vocabulary simply with the set of all bytes (0-255).
        """
        self.vocab = {byte_value: bytes([byte_value]) for byte_value in range(256)}

    def _finalize_vocab(self):
        """
        Finalize the vocabulary by adding special tokens.
        """
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

    def _update_counter(
        self,
        most_frequent_pair: tuple[bytes, bytes],
        token_counter: list[tuple[list[bytes], int]],
        pair_counter: Counter,
        pair_indexes: defaultdict[Counter],
        merged_bytes: bytes,
    ) -> Counter:
        """
        Update all counters after merging a pair of tokens.
        Returns new pairs that were created by the merge.
        """
        affected_indexes: Counter = pair_indexes[most_frequent_pair].copy()
        new_pair_counter = Counter()

        for index, index_count in affected_indexes.items():
            if index_count == 0:  # skip if the pair is no longer present in this token
                continue

            tokens, count = token_counter[index]
            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == most_frequent_pair:
                    # remove the old pair
                    pair_counter[(tokens[i], tokens[i + 1])] -= count
                    pair_indexes[(tokens[i], tokens[i + 1])][index] -= 1

                    if i > 0:
                        # remove the old left pair
                        old_left_pair = (tokens[i - 1], tokens[i])
                        pair_counter[old_left_pair] -= count
                        pair_indexes[old_left_pair][index] -= 1
                        # add the new left pair
                        new_left_pair = (tokens[i - 1], merged_bytes)
                        new_pair_counter[new_left_pair] += count
                        pair_indexes[new_left_pair][index] += 1

                    if i + 1 < len(tokens) - 1:
                        # remove the old right pair
                        old_right_pair = (tokens[i + 1], tokens[i + 2])
                        pair_counter[old_right_pair] -= count
                        pair_indexes[old_right_pair][index] -= 1
                        # add the new right pair
                        new_right_pair = (merged_bytes, tokens[i + 2])
                        new_pair_counter[new_right_pair] += count
                        pair_indexes[new_right_pair][index] += 1
                    new_tokens.append(merged_bytes)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            if i < len(tokens):  # append the last token if not merged
                new_tokens.append(tokens[i])
            token_counter[index] = (new_tokens, count)

        return new_pair_counter

    def train(self, input_path: str | os.PathLike):
        """
        Train the BPE model on the given input file.
        """
        self._init_vocab()

        token_counter = pretokenize_file(input_path, self.special_tokens)
        # convert tuples to lists for mutability and easier indexing
        token_counter = [
            ([bytes([b]) for b in token], count)
            for token, count in token_counter.items()
        ]

        # keep track of pair count, (token1, token2) -> count
        pair_counter = Counter()
        # (token1, token2) -> index in token_counter -> count(may have duplicates in one token)
        pair_indexes = defaultdict(Counter)

        for index, (tokens, count) in enumerate(token_counter):
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counter[pair] += count
                pair_indexes[pair][index] += 1  # count of this pair in this token

        heap = [
            PairItem(token1, token2, count)
            for (token1, token2), count in pair_counter.items()
        ]
        heapq.heapify(heap)  # make heap in-place

        merge_count = self.vocab_size - len(self.vocab) - len(self.special_tokens)

        while merge_count > 0:
            if not heap:
                break

            most_frequent_item = None
            while heap:
                candidate = heapq.heappop(heap)
                candidate_pair = (candidate.token1, candidate.token2)
                # check if the candidate is still valid
                if pair_counter[candidate_pair] == candidate.count:
                    most_frequent_item = candidate
                    break

                # update the invalid candidate's count
                candidate.count = pair_counter[candidate_pair]
                heapq.heappush(heap, candidate)

            if most_frequent_item is None:
                break

            most_frequent_pair = (most_frequent_item.token1, most_frequent_item.token2)
            merged_bytes = most_frequent_pair[0] + most_frequent_pair[1]

            # update the counters and get the new pairs counter
            new_pair_counter = self._update_counter(
                most_frequent_pair,
                token_counter,
                pair_counter,
                pair_indexes,
                merged_bytes,
            )
            # update the pair counter globally and push new pairs to the heap
            for new_pair, new_count in new_pair_counter.items():
                pair_counter[new_pair] += new_count
                pair_item = PairItem(new_pair[0], new_pair[1], pair_counter[new_pair])
                heapq.heappush(heap, pair_item)

            # only add unique tokens to vocab to avoid duplicates.
            # different pairs may produce the same merged result, especially with repetitive characters
            # e.g., (b'rr', b'r') and (b'r', b'rr') both produce b'rrr'.
            # ensure each unique token gets only one ID.
            if merged_bytes in self.vocab_set:
                continue
            self.vocab_set.add(merged_bytes)

            new_token_id = len(self.vocab)
            self.vocab[new_token_id] = merged_bytes
            self.merges.append(most_frequent_pair)
            merge_count -= 1

        self._finalize_vocab()

        return self.vocab, self.merges

    def to_files(self, vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
        """
        Save the trained vocabulary and merges to files.
        """
        gpt2_byte_encoder = gpt2_bytes_to_unicode()
        # save vocabulary as JSON: {"token": id}
        vocab_dict = {}
        print(
            f"vocab size: {len(self.vocab.values())}, set vocab size: {len(set(self.vocab.values()))}"
        )
        for token_id, token_bytes in self.vocab.items():
            token_str = "".join(gpt2_byte_encoder[b] for b in token_bytes)
            vocab_dict[token_str] = token_id

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # save merges as text file: "token1 token2" per line

        with open(merges_path, "w", encoding="utf-8") as f:
            for token1_bytes, token2_bytes in self.merges:
                token1_str = "".join(gpt2_byte_encoder[b] for b in token1_bytes)
                token2_str = "".join(gpt2_byte_encoder[b] for b in token2_bytes)

                f.write(f"{token1_str} {token2_str}\n")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.
    Returns the vocabulary and merges.
    """
    trainer = BPETrainer(vocab_size, special_tokens)
    return trainer.train(input_path)
