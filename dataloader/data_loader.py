"""
Utilities to build vocabulary and PyTorch DataLoaders for Student Feedback.
"""

from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(
    ds_dict: DatasetDict,
    min_freq: int = 2,
    split_name: str = "train",
    text_key: str = "sentence",
) -> Dict[str, int]:
    """
    Build token-to-index vocabulary from a dataset split.

    Args:
        ds_dict: HuggingFace DatasetDict with train/validation/test splits.
        min_freq: Minimum word frequency to include in vocabulary.
        split_name: Split name used to build vocabulary.
        text_key: Text column name.

    Returns:
        Vocabulary mapping from token to integer index.
    """
    if split_name not in ds_dict:
        raise KeyError(f"Split '{split_name}' not found in DatasetDict.")

    counter = Counter()
    for example in ds_dict[split_name]:
        counter.update(example[text_key].lower().split())

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def make_collate_fn(
    vocab: Dict[str, int],
    text_key: str = "sentence",
    sentiment_key: str = "sentiment",
    topic_key: str = "topic",
) -> Callable:
    """
    Create a collate function that tokenizes and pads a batch.

    Args:
        vocab: Token-to-index mapping.
        text_key: Text field name.
        sentiment_key: Sentiment label field name.
        topic_key: Topic label field name.

    Returns:
        A collate function compatible with torch DataLoader.
    """

    def collate_fn(batch):
        sentences = []
        sentiments = []
        topics = []
        lengths = []

        for item in batch:
            tokens = [vocab.get(word.lower(), UNK_IDX) for word in item[text_key].split()]
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            sentences.append(token_tensor)
            lengths.append(len(tokens))
            sentiments.append(item[sentiment_key])
            topics.append(item[topic_key])

        sentences_padded = pad_sequence(
            sentences,
            batch_first=True,
            padding_value=PAD_IDX,
        )
        attention_mask = (sentences_padded != PAD_IDX).long()

        return {
            "input_ids": sentences_padded,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "attention_mask": attention_mask,
            "sentiment": torch.tensor(sentiments, dtype=torch.long),
            "topic": torch.tensor(topics, dtype=torch.long),
        }

    return collate_fn


def get_dataloaders(
    ds_dict: DatasetDict,
    vocab: Dict[str, int],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for available splits.

    Args:
        ds_dict: HuggingFace DatasetDict with splits.
        vocab: Token-to-index mapping.
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes for DataLoader.
        pin_memory: Whether to pin memory in DataLoader.

    Returns:
        Dictionary mapping split names to DataLoader instances.
    """
    collate_fn = make_collate_fn(vocab=vocab)

    loaders: Dict[str, DataLoader] = {}

    if "train" in ds_dict:
        loaders["train"] = DataLoader(
            ds_dict["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if "validation" in ds_dict:
        loaders["validation"] = DataLoader(
            ds_dict["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if "test" in ds_dict:
        loaders["test"] = DataLoader(
            ds_dict["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders


def build_vocab_and_loaders(
    ds_dict: DatasetDict,
    min_freq: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dict[str, int], Dict[str, DataLoader]]:
    """
    Convenience helper to create vocab and all available DataLoaders.

    Args:
        ds_dict: HuggingFace DatasetDict with splits.
        min_freq: Minimum token frequency for vocabulary.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory in DataLoaders.

    Returns:
        Tuple of (vocab, loaders).
    """
    vocab = build_vocab(ds_dict=ds_dict, min_freq=min_freq)
    loaders = get_dataloaders(
        ds_dict=ds_dict,
        vocab=vocab,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return vocab, loaders