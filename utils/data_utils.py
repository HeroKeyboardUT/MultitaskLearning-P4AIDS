"""
Data processing utilities for Student Feedback dataset.

This module provides functions to download, process, and create HuggingFace
datasets from Google Drive.
"""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value


def read_file(
    file_path: str,
    encoding: str = "utf-8",
    as_int: bool = False
) -> List:
    """
    Read lines from a text file.

    Args:
        file_path: Path to the text file
        encoding: File encoding (default: utf-8)
        as_int: If True, convert each line to integer

    Returns:
        List of lines (as strings or integers)
    """
    with open(file_path, "r", encoding=encoding) as f:
        lines = [line.strip() for line in f]

    if as_int:
        lines = [int(line) for line in lines]

    return lines


def filter_and_remap(
    sentences: List[str],
    sentiments: List[int],
    topics: List[int],
    exclude_sentiments: List[int] = None,
    exclude_topics: List[int] = None,
    sentiment_mapping: Dict[int, int] = None,
    topic_mapping: Dict[int, int] = None
) -> pd.DataFrame:
    """
    Create a dataframe and apply filtering and label mapping.

    Args:
        sentences: List of sentence texts
        sentiments: List of sentiment labels
        topics: List of topic labels
        exclude_sentiments: List of sentiment values to exclude
        exclude_topics: List of topic values to exclude
        sentiment_mapping: Dictionary to remap sentiment labels
        topic_mapping: Dictionary to remap topic labels

    Returns:
        Filtered and processed pandas DataFrame
    """
    # Create dataframe
    df = pd.DataFrame({
        "sentence": sentences,
        "sentiment": sentiments,
        "topic": topics
    })

    # Filter out unwanted labels
    if exclude_sentiments:
        df = df[~df["sentiment"].isin(exclude_sentiments)].copy()
    if exclude_topics:
        df = df[~df["topic"].isin(exclude_topics)].copy()

    # Remap labels
    if sentiment_mapping:
        df["sentiment"] = df["sentiment"].replace(sentiment_mapping)
    if topic_mapping:
        df["topic"] = df["topic"].replace(topic_mapping)

    return df.reset_index(drop=True)


def process_split(
    split_name: str,
    data_dir: str,
    file_name_patterns: Dict[str, str],
    exclude_sentiments: List[int] = None,
    exclude_topics: List[int] = None,
    sentiment_mapping: Dict[int, int] = None,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Read local data files for a split and return processed dataframe.

    Args:
        split_name: Name of the split (train, validation, test)
        data_dir: Directory containing local data files
        file_name_patterns: Filename patterns for different data types
        exclude_sentiments: List of sentiment labels to exclude
        exclude_topics: List of topic labels to exclude
        sentiment_mapping: Dictionary to remap sentiment labels
        encoding: File encoding

    Returns:
        Processed pandas DataFrame for the split
    """
    print(f"--- Loading split: {split_name} ---")

    resolved_patterns = {
        key: pattern.format(split=split_name)
        for key, pattern in file_name_patterns.items()
    }

    # Build full paths
    sentences_path = os.path.join(data_dir, resolved_patterns["sentences"])
    sentiments_path = os.path.join(data_dir, resolved_patterns["sentiments"])
    topics_path = os.path.join(data_dir, resolved_patterns["topics"])

    # Read data
    sentences = read_file(sentences_path, encoding=encoding)
    sentiments = read_file(sentiments_path, encoding=encoding, as_int=True)
    topics = read_file(topics_path, encoding=encoding, as_int=True)

    # Create and filter dataframe
    df = filter_and_remap(
        sentences=sentences,
        sentiments=sentiments,
        topics=topics,
        exclude_sentiments=exclude_sentiments,
        exclude_topics=exclude_topics,
        sentiment_mapping=sentiment_mapping
    )

    return df


def download_split(
    split_name: str,
    file_ids_by_split: Dict[str, Dict[str, str]],
    data_dir: str,
    file_name_patterns: Dict[str, str],
    quiet: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Download raw files for one split from Google Drive.

    Args:
        split_name: Name of the split (train, validation, test)
        file_ids_by_split: Dictionary of split names to file ID dictionaries
        data_dir: Directory to save downloaded files
        file_name_patterns: Filename patterns for different data types
        quiet: If True, suppress gdown output
        overwrite: If True, re-download even if local files already exist
    """
    import gdown

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    split_file_ids = file_ids_by_split[split_name]
    resolved_patterns = {
        key: pattern.format(split=split_name)
        for key, pattern in file_name_patterns.items()
    }

    target_paths = {
        "sentences": os.path.join(data_dir, resolved_patterns["sentences"]),
        "sentiments": os.path.join(data_dir, resolved_patterns["sentiments"]),
        "topics": os.path.join(data_dir, resolved_patterns["topics"]),
    }

    for key, target_path in target_paths.items():
        if not overwrite and os.path.exists(target_path):
            continue
        gdown.download(id=split_file_ids[key], output=target_path, quiet=quiet)


def download_data(
    file_ids_by_split: Dict[str, Dict[str, str]],
    splits: List[str],
    data_dir: str,
    file_name_patterns: Dict[str, str],
    quiet: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Download raw files for all requested splits from Google Drive.

    Args:
        file_ids_by_split: Dictionary of split names to file ID dictionaries
        splits: List of split names to download
        data_dir: Directory to save downloaded files
        file_name_patterns: Filename patterns for different data types
        quiet: If True, suppress gdown output
        overwrite: If True, re-download even if local files already exist
    """
    for split_name in splits:
        print(f"--- Downloading split: {split_name} ---")
        download_split(
            split_name=split_name,
            file_ids_by_split=file_ids_by_split,
            data_dir=data_dir,
            file_name_patterns=file_name_patterns,
            quiet=quiet,
            overwrite=overwrite,
        )


def build_dataset(
    splits_data: Dict[str, pd.DataFrame],
    sentiment_names: List[str],
    topic_names: List[str]
) -> DatasetDict:
    """
    Create a HuggingFace DatasetDict from processed dataframes.

    Args:
        splits_data: Dictionary of split names to processed dataframes
        sentiment_names: List of sentiment label names
        topic_names: List of topic label names

    Returns:
        HuggingFace DatasetDict with all splits
    """
    # Define dataset features
    features = Features({
        "sentence": Value("string"),
        "sentiment": ClassLabel(names=sentiment_names),
        "topic": ClassLabel(names=topic_names),
    })

    # Create datasets for each split
    dataset_dict = {}
    for split_name, df in splits_data.items():
        dataset_dict[split_name] = Dataset.from_pandas(
            df,
            features=features,
            preserve_index=False
        )

    return DatasetDict(dataset_dict)


def load_dataset(
    split_name: str,
    data_dir: str,
    file_name_patterns: Dict[str, str],
    exclude_sentiments: List[int] = None,
    exclude_topics: List[int] = None,
    sentiment_mapping: Dict[int, int] = None,
    sentiment_names: List[str] = None,
    topic_names: List[str] = None,
    encoding: str = "utf-8"
) -> Dataset:
    """
    Load and preprocess local files for one split.

    Args:
        split_name: Name of the split to process (train, validation, test)
        data_dir: Directory containing local data files
        file_name_patterns: Filename patterns for different data types
        exclude_sentiments: List of sentiment labels to exclude
        exclude_topics: List of topic labels to exclude
        sentiment_mapping: Dictionary to remap sentiment labels
        sentiment_names: List of sentiment label names
        topic_names: List of topic label names
        encoding: File encoding

    Returns:
        HuggingFace Dataset for the requested split
    """
    df = process_split(
        split_name=split_name,
        data_dir=data_dir,
        file_name_patterns=file_name_patterns,
        exclude_sentiments=exclude_sentiments,
        exclude_topics=exclude_topics,
        sentiment_mapping=sentiment_mapping,
        encoding=encoding
    )

    features = Features({
        "sentence": Value("string"),
        "sentiment": ClassLabel(names=sentiment_names or ["negative", "positive"]),
        "topic": ClassLabel(names=topic_names or ["lecturer", "training_program", "facility"]),
    })

    return Dataset.from_pandas(df, features=features, preserve_index=False)


def load_splits(
    splits: List[str],
    data_dir: str,
    file_name_patterns: Dict[str, str],
    exclude_sentiments: List[int] = None,
    exclude_topics: List[int] = None,
    sentiment_mapping: Dict[int, int] = None,
    sentiment_names: List[str] = None,
    topic_names: List[str] = None,
    encoding: str = "utf-8",
) -> DatasetDict:
    """
    Load and preprocess multiple splits into a DatasetDict.

    Args:
        splits: Split names to load
        data_dir: Directory containing local data files
        file_name_patterns: Filename patterns for different data types
        exclude_sentiments: List of sentiment labels to exclude
        exclude_topics: List of topic labels to exclude
        sentiment_mapping: Dictionary to remap sentiment labels
        sentiment_names: List of sentiment label names
        topic_names: List of topic label names
        encoding: File encoding

    Returns:
        DatasetDict with requested splits.
    """
    loaded = {}
    for split_name in splits:
        loaded[split_name] = load_dataset(
            split_name=split_name,
            data_dir=data_dir,
            file_name_patterns=file_name_patterns,
            exclude_sentiments=exclude_sentiments,
            exclude_topics=exclude_topics,
            sentiment_mapping=sentiment_mapping,
            sentiment_names=sentiment_names,
            topic_names=topic_names,
            encoding=encoding,
        )

    return DatasetDict(loaded)


def plot_distribution(
    ds_dict: DatasetDict,
    column_name: str,
    title: str,
    images_dir: str,
    file_name: str = None,
) -> str:
    """
    Plot label distribution by split and save the chart to disk.

    Args:
        ds_dict: DatasetDict containing train/validation/test splits
        column_name: Target label column name (e.g., sentiment, topic)
        title: Chart title suffix
        images_dir: Output directory for saved image
        file_name: Optional output file name (PNG). Auto-generated if None.

    Returns:
        Absolute path of the saved image.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not ds_dict:
        raise ValueError("ds_dict is empty.")

    plot_data = []
    for split_name in ds_dict.keys():
        df_temp = ds_dict[split_name].to_pandas()
        df_temp["split"] = split_name
        plot_data.append(df_temp)

    df_all = pd.concat(plot_data, ignore_index=True)

    # Convert numeric labels to label names using ClassLabel metadata.
    ref_split = "train" if "train" in ds_dict else next(iter(ds_dict.keys()))
    feature = ds_dict[ref_split].features[column_name]
    if hasattr(feature, "int2str"):
        df_all[column_name] = df_all[column_name].apply(lambda x: feature.int2str(int(x)))

    Path(images_dir).mkdir(parents=True, exist_ok=True)
    output_name = file_name or f"{column_name}_distribution.png"
    output_path = os.path.join(images_dir, output_name)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df_all, x=column_name, hue="split", palette="magma")

    plt.title(f"Label Distribution: {title}", fontsize=16, fontweight="bold")
    plt.xlabel("Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Split")

    for patch in ax.patches:
        if patch.get_height() > 0:
            ax.annotate(
                f"{int(patch.get_height())}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                xytext=(0, 7),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return os.path.abspath(output_path)


def visualize_data(ds_dict: DatasetDict, images_dir: str = "images") -> Dict[str, str]:
    """
    Generate and save default distribution plots for sentiment and topic.

    Args:
        ds_dict: DatasetDict containing multiple splits
        images_dir: Output directory for saved charts

    Returns:
        Mapping from plot key to saved image path.
    """
    saved_paths = {
        "sentiment": plot_distribution(
            ds_dict=ds_dict,
            column_name="sentiment",
            title="Sentiment",
            images_dir=images_dir,
            file_name="sentiment_distribution.png",
        ),
        "topic": plot_distribution(
            ds_dict=ds_dict,
            column_name="topic",
            title="Topic",
            images_dir=images_dir,
            file_name="topic_distribution.png",
        ),
    }

    return saved_paths


def get_class_weights(
    train_dataset: Dataset,
    column_name: str,
    num_classes: int = None,
):
    """
    Compute balanced class weights from train split only.

    Args:
        train_dataset: HuggingFace Dataset of train split
        column_name: Target label column name (e.g., sentiment, topic)

    Args:
        num_classes: Total number of classes. If provided, output weights
            always have this length (useful when some classes are absent
            in the current split).

    Returns:
        Torch tensor of class weights sorted by class id.
    """
    import torch

    labels = torch.tensor(train_dataset[column_name], dtype=torch.long)

    if labels.numel() == 0:
        raise ValueError(f"No labels found in column '{column_name}'.")

    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))

    return weights
