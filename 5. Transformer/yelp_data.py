from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _tokenize_split(split, tokenizer, max_len, batch_size):
    def encode(batch):
        return {"input_ids": tokenizer(batch["text"], truncation=True,
                                       max_length=max_len)["input_ids"]}

    return split.select_columns(["text", "label"]).map(
        encode, batched=True, batch_size=batch_size, remove_columns=["text"]
    )


def _build_pruned_tokenizer(train, directory, max_len, rare_token_limit, batch_size):
    original = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def encode(batch):
        ids = original(batch["text"], truncation=True,
                       max_length=max_len)["input_ids"]
        return {"input_ids": ids}

    tokens = train.select_columns(["text"]).map(
        encode, batched=True, batch_size=batch_size, remove_columns=["text"]
    )
    counts = np.zeros(len(original), dtype=np.int64)
    for batch in tokens.iter(batch_size=batch_size):
        counts += np.bincount(
            np.concatenate(batch["input_ids"]).astype(np.int64),
            minlength=len(original),
        )

    specials = set(original.all_special_ids)
    vocab = original.get_vocab()
    pieces = [piece for piece, index in sorted(vocab.items(), key=lambda pair: pair[1])
              if index in specials or counts[index] > rare_token_limit]
    backend = Tokenizer(WordPiece(vocab={piece: i for i, piece in enumerate(pieces)},
                                  unk_token="[UNK]"))
    backend.normalizer = BertNormalizer(lowercase=True)
    backend.pre_tokenizer = BertPreTokenizer()
    backend.decoder = WordPieceDecoder()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", backend.token_to_id("[CLS]")),
                        ("[SEP]", backend.token_to_id("[SEP]"))],
    )
    pruned = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=max_len,
    )
    pruned.save_pretrained(directory)
    return pruned


def load_yelp_data(max_len=512, rare_token_limit=5, batch_size=32,
                   tokenize_batch_size=2048, directory=None):
    if directory is None:
        directory = Path(__file__).resolve().parent / "legacy" / (
            f"yelp_bert_pruned_tokenizer_len{max_len}_rare{rare_token_limit}"
        )
    directory = Path(directory)
    train_cache, test_cache = directory / "train_tokens", directory / "test_tokens"

    if (directory / "tokenizer.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True)
    else:
        dataset = load_dataset("ajay232/yelp_polarity")
        tokenizer = _build_pruned_tokenizer(
            dataset["train"], directory, max_len, rare_token_limit, tokenize_batch_size
        )

    if train_cache.is_dir() and test_cache.is_dir():
        train_tokens = load_from_disk(str(train_cache))
        test_tokens = load_from_disk(str(test_cache))
    else:
        if "dataset" not in locals():
            dataset = load_dataset("ajay232/yelp_polarity")
        train_tokens = _tokenize_split(dataset["train"], tokenizer, max_len, tokenize_batch_size)
        test_tokens = _tokenize_split(dataset["test"], tokenizer, max_len, tokenize_batch_size)
        if not train_cache.exists():
            train_tokens.save_to_disk(str(train_cache))
        if not test_cache.exists():
            test_tokens.save_to_disk(str(test_cache))

    pad_id = tokenizer.pad_token_id

    def collate(examples):
        width = max(len(example["input_ids"]) for example in examples)
        ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        for row, example in enumerate(examples):
            ids[row, :len(example["input_ids"])] = torch.tensor(example["input_ids"])
        return ids, labels

    train_loader = DataLoader(train_tokens, batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_tokens, batch_size=batch_size, shuffle=False,
                             collate_fn=collate)
    return tokenizer, train_loader, test_loader
