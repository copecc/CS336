import os
import torch
import json
import random

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from cs336_alignment.common import ALPACA_SFT_PROMPT_PATH


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_path: str, seq_length: int, shuffle: bool = False):
        # read examples of (prompt, response) pairs
        with open(dataset_path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        if shuffle:
            random.shuffle(examples)
        # Alpaca SFT prompt
        with open(ALPACA_SFT_PROMPT_PATH, "r", encoding="utf-8") as f:
            alpaca_sft_prompt = f.read().strip()
            alpaca_sft_prompt = "<|begin_of_text|>" + alpaca_sft_prompt + "<|end_of_text|>"

        documents = [
            alpaca_sft_prompt.format(instruction=example["prompt"], response=example["response"])
            for example in examples
        ]
        full_text = "".join(documents)

        all_token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        batch_size = len(all_token_ids) // seq_length
        self.input_ids = all_token_ids[: batch_size * seq_length]
        self.labels = (all_token_ids + [tokenizer.eos_token_id])[1 : batch_size * seq_length + 1]

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long).reshape(batch_size, seq_length)
        self.labels = torch.tensor(self.labels, dtype=torch.long).reshape(batch_size, seq_length)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
