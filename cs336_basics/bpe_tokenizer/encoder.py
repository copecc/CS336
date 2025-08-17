import multiprocessing as mp
from random import sample
import regex as re
import numpy as np
import os

from loguru import logger
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.pretokenization import find_chunk_boundaries
from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer
from cs336_basics.bpe_tokenizer.trainer import BPETrainer


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = None,
    save: bool = False,
    vocab_path: str | os.PathLike = None,
    merge_path: str | os.PathLike = None,
    desired_num_chunks: int = 8,
) -> BPETokenizer:
    """
    Train a BPE tokenizer on the given text file.
    """
    special_tokens = special_tokens or ["<|endoftext|>"]

    logger.info(
        f"Training BPE tokenizer on {input_path}, vocab size: {vocab_size}, special tokens: {special_tokens}, num chunks: {desired_num_chunks}"
    )

    trainer = BPETrainer(vocab_size, special_tokens)
    vocab, merges = trainer.train(input_path, desired_num_chunks)

    logger.info(f"Trained BPE tokenizer with vocab size: {len(vocab)}")

    if save:
        trainer.to_files(vocab_path, merge_path)

    return BPETokenizer(vocab, merges)


def load_bpe_tokenizer(
    vocab_path: str | os.PathLike,
    merge_path: str | os.PathLike,
    special_tokens: list[str] = None,
):
    """
    Load a BPE tokenizer from the given vocab and merge files.
    """
    special_tokens = special_tokens or ["<|endoftext|>"]
    return BPETokenizer.from_files(vocab_path, merge_path, special_tokens)


def tokenize_chunk(input_path, start, end, tokenizer: BPETokenizer) -> list[int]:
    """
    Tokenize a chunk of text from the input file.
    """
    logger.debug(
        f"Tokenizing bytes {start}-{end}, total {(end - start) / (1024 * 1024):.2f} MB..."
    )

    ids = []
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        ids.extend(tokenizer.encode(chunk))

    logger.debug(f"Tokenized chunk {start}-{end}, got {len(ids)} tokens")

    return ids


def encode_file(
    input_path: str,
    tokenizer: BPETokenizer,
    output_file: str,
    use_memmap: bool = False,
    desired_num_chunks: int = 8,
) -> None:
    """
    Encode a text file into a sequence of token IDs.
    """

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
    # safe to use same tokenizer since encode() is stateless
    chunk_tasks = [
        (input_path, start, end, tokenizer)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    logger.info(f"Encoding file {input_path} using {desired_num_chunks} chunks...")

    with mp.Pool(processes=min(desired_num_chunks, 4 * mp.cpu_count() // 5)) as pool:
        results = pool.starmap(tokenize_chunk, chunk_tasks)
    logger.info(f"Encoding completed. Saving to {output_file}...")

    total_tokens = sum(len(result) for result in results)
    if use_memmap:
        token_array = np.memmap(output_file, np.uint16, "w+", shape=(total_tokens,))
    else:
        token_array = np.empty(total_tokens, dtype=np.uint16)

    offset = 0
    for result in tqdm(results):
        token_array[offset : offset + len(result)] = result
        offset += len(result)

    if use_memmap:
        token_array.flush()
    else:
        np.save(output_file, token_array)

    logger.success(f"Saved token array to {output_file}")


def encode_file_streaming(
    input_path: str,
    tokenizer: BPETokenizer,
    output_file: str,
    desired_MB_per_chunk: int = 200,
) -> None:
    """
    Encode a text file into a sequence of token IDs using streaming.
    This can be more memory efficient and faster for large files.
    """

    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        desired_num_chunks = max(1, file_size // (desired_MB_per_chunk * 1024 * 1024))
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    with open(output_file, "wb"):
        pass  # clear the output file

    # safe to use same tokenizer since encode() is stateless
    chunk_tasks = [
        (input_path, start, end, tokenizer)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    logger.info(f"Encoding file {input_path} using {desired_num_chunks} chunks...")

    processes = min(desired_num_chunks, 4 * mp.cpu_count() // 5)
    batch_size = min(len(chunk_tasks), 2 * processes)

    with mp.Pool(processes=processes) as pool:
        for i in range(0, len(chunk_tasks), batch_size):
            batch_tasks = chunk_tasks[i : i + batch_size]
            logger.debug(
                f"Processing chunks {i} to {i + len(batch_tasks)} / {len(chunk_tasks)}..."
            )

            results = pool.starmap(tokenize_chunk, batch_tasks)
            for result in results:
                with open(output_file, "ab") as out_f:
                    np_array = np.array(result, dtype=np.uint16)
                    np_array.tofile(out_f)

    logger.success(f"Saved token array to {output_file}")


def encode_file_naive(
    input_path: str, tokenizer: BPETokenizer, output_file: str, use_memmap: bool = False
) -> None:

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Encoding file {input_path} serially...")

    token_ids = tokenizer.encode(text)

    logger.info(f"Encoding completed. Saving to {output_file}...")

    if use_memmap:
        token_array = np.memmap(
            output_file, dtype=np.uint16, mode="w+", shape=(len(token_ids),)
        )
    else:
        token_array = np.empty(len(token_ids), dtype=np.uint16)

    token_array[:] = token_ids

    if use_memmap:
        token_array.flush()
    else:
        np.save(output_file, token_array)

    logger.success(f"Saved token array to {output_file}")


def sample_from_chunk(
    input_path, start, end, sample_size: int, special_tokens: list[str]
) -> list[str]:
    """
    Sample documents from a chunk of text.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    special_tokens = special_tokens or [r"?!"]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    # split the chunk into segments using the special tokens
    special_pattern = re.compile(f'{"|".join(map(re.escape, special_tokens))}')
    segments = special_pattern.split(chunk)
    # in case have fewer segments than sample_size
    if len(segments) <= sample_size:
        return segments
    return sample(segments, sample_size)


def sample_from_file(
    input_path: str,
    sample_size: int,
    special_tokens: list[str],
    desired_num_chunks: int = 8,
) -> list[str]:
    """
    Sample documents from a text file.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    chunk_tasks = [
        (input_path, start, end, sample_size, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    # sample from each chunk
    with mp.Pool(processes=min(desired_num_chunks, 4 * mp.cpu_count() // 5)) as pool:
        results = pool.starmap(sample_from_chunk, chunk_tasks)
    # flatten the list of samples, and return a random sample
    samples = [item for sublist in results for item in sublist]
    if len(samples) <= sample_size:
        return samples
    return sample(samples, sample_size)
