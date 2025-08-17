import os
import regex as re
import multiprocessing as mp

from collections import Counter
from typing import BinaryIO, Iterator


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_iter(text, special_tokens: list[str]) -> Iterator[bytes]:
    """
    Returns an iterator over the pre-tokenized text.
    """
    special_tokens = special_tokens or [r"?!"]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    # using '(...|...)' to match any of the special tokens and capture them
    special_pattern = re.compile(f'({"|".join(map(re.escape, special_tokens))})')
    gpt2_split_pattern = re.compile(PAT)

    for segment in special_pattern.split(text):
        if segment in special_tokens:
            yield segment.encode("utf-8")
            continue
        for match in gpt2_split_pattern.finditer(segment):
            if matched_text := match.group():
                yield matched_text.encode("utf-8")


def pretokenize_text(text, special_tokens: list[str]) -> list[bytes]:
    """
    Pre-tokenize the input text into sub-word tokens.
    """
    return list(pretokenize_iter(text, special_tokens))


def pretokenize_chunk_to_counter(input_path, start, end, special_tokens: list[str]) -> Counter:
    """
    Pre-tokenize a chunk of text into sub-word tokens, returning a token frequency counter excluding special tokens.
    """
    special_tokens_bytes = {st.encode("utf-8") for st in special_tokens}
    pretoken_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        for pretoken_bytes in pretokenize_iter(chunk, special_tokens):
            if pretoken_bytes in special_tokens_bytes:  # skip special tokens
                continue
            pretoken_counter[tuple(pretoken_bytes)] += 1

    return pretoken_counter


def pretokenize_file_to_counter(input_path, special_tokens: list[str], desired_num_chunks: int = 8) -> Counter:
    """
    Pre-tokenize the entire input file.
    """

    pretoken_counter = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    chunk_tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(processes=min(desired_num_chunks, 4 * mp.cpu_count() // 5)) as pool:
        for result in pool.starmap(pretokenize_chunk_to_counter, chunk_tasks):
            pretoken_counter += result

    return pretoken_counter
