import shutil
import mmh3
import os
import regex as re
import unicodedata

from collections import Counter, defaultdict


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    line_counts = Counter(
        mmh3.hash(line, signed=False) for input_file in input_files for line in open(input_file, "r", encoding="utf-8")
    )

    for input_file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                hash_val = mmh3.hash(line, signed=False)
                if line_counts[hash_val] == 1:  # Only write unique lines
                    f_out.write(line)


def normalize_text(text: str) -> str:
    """
    Normalize the input text by applying NFD unicode normalization,
    removing diacritics, lowercasing, and normalizing whitespace.
    """
    # NFD unicode normalization
    text = unicodedata.normalize("NFD", text)
    # Removing diacritics (combining marks)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalizing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def minhash(words_ngrams: list[str], num_hashes: int) -> list[int]:
    """
    Compute MinHash signatures for the given n-grams.
    Return the MinHash signatures of len <num_hashes>.
    """
    minhashes = [min(mmh3.hash(ngram, seed=i, signed=False) for ngram in words_ngrams) for i in range(num_hashes)]
    return minhashes


def split_into_bands(signature: list[int], num_bands: int) -> list[tuple[int]]:
    """
    Split the MinHash signature into bands for LSH.
    """
    band_size = len(signature) // num_bands
    return [tuple(signature[i * band_size : (i + 1) * band_size]) for i in range(num_bands)]


class UnionFind:
    """
    Union-Find data structure for tracking connected components.
    """

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            self.parent[fy] = fx


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    Perform MinHash-based deduplication on the given input files.
    """
    doc_ngrams: list[list[str]] = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())
            words = text.split()
            words_ngrams = [" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1)]
            doc_ngrams.append(words_ngrams)

    minhash_signatures = [minhash(ngrams, num_hashes) for ngrams in doc_ngrams]
    banded_signatures = [split_into_bands(signature, num_bands) for signature in minhash_signatures]
    # Create buckets for potential collisions
    buckets = defaultdict(list)
    for doc_id, bands in enumerate(banded_signatures):
        for band_idx, band in enumerate(bands):
            # map band to doc_id for potential collisions
            buckets[(band_idx, band)].append(doc_id)
    # If a band has multiple documents, consider all pairs
    candidate_pairs = set()
    for doc_ids in buckets.values():
        if len(doc_ids) <= 1:
            continue
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                candidate_pairs.add(tuple(sorted((doc_ids[i], doc_ids[j]))))
    # Compute Jaccard similarity for candidate pairs, clustering similar documents
    uf = UnionFind(len(doc_ngrams))
    for doc1, doc2 in candidate_pairs:
        set1 = set(doc_ngrams[doc1])
        set2 = set(doc_ngrams[doc2])
        jaccard = len(set1 & set2) / len(set1 | set2)
        if jaccard >= jaccard_threshold:
            uf.union(doc1, doc2)
    # Write clustered documents to output files
    for doc_id in range(len(doc_ngrams)):
        root = uf.find(doc_id)
        # Avoid writing duplicates; only write one document per cluster
        if root != doc_id:
            continue
        input_file = input_files[doc_id]
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        shutil.copyfile(input_file, output_file)
