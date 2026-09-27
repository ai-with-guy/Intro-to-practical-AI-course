"""Importable batch collation for Windows-spawned notebook DataLoader workers."""

import torch


class TokenBatchCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        width = max(len(example["input_ids"]) for example in examples)
        batch = torch.full((len(examples), width), self.pad_id, dtype=torch.long)
        for row, example in enumerate(examples):
            ids = example["input_ids"]
            batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return batch
