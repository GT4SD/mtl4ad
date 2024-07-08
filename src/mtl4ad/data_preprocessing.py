"""Dataset Utilities: Loading, Preprocessing, and Filtering"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataset_from_folders(base_path: Union[str, Path]) -> DatasetDict:
    """Load and concatenate datasets from multiple folders.

    Args:
        base_path: The base path to the dataset folders.

    Returns:
        The concatenated dataset with 'train' and 'validation' splits.
    """
    all_train_datasets = []
    all_valid_datasets = []

    base_path = Path(base_path) if isinstance(base_path, str) else base_path
    for folder in base_path.iterdir():
        if folder.is_dir():
            train_file = folder / "train.parquet"
            valid_file = folder / "validation.parquet"

            if train_file.exists():
                train_dataset = load_dataset(
                    "parquet", data_files=str(train_file), split="train", num_proc=32
                )
                all_train_datasets.append(train_dataset)

            if valid_file.exists():
                valid_dataset = load_dataset(
                    "parquet", data_files=str(valid_file), split="train", num_proc=32
                )
                all_valid_datasets.append(valid_dataset)

    train_dataset = concatenate_datasets(all_train_datasets) if all_train_datasets else None
    valid_dataset = concatenate_datasets(all_valid_datasets) if all_valid_datasets else None

    dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset})

    return dataset


def preprocess(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: Optional[int]
) -> Dataset:
    """
    Preprocesses the dataset by formatting the examples and tokenizing them.

    Args:
        dataset: The dataset to preprocess.
        tokenizer: The tokenizer to use for processing the dataset.
        max_length: The maximum length of the tokenized sequences.

    Returns:
        The preprocessed dataset.
    """

    def process(examples: dict) -> dict:
        """
        Helper function to process each example in the dataset.

        Args:
            examples: A batch of examples from the dataset.

        Returns:
            A dictionary with tokenized inputs and labels.
        """
        formatted_prompts = [
            f"{source} \n\n {target}"
            for source, target in zip(examples["source"], examples["target"])
        ]
        model_inputs = tokenizer(
            formatted_prompts,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(process, batched=True)


def filter_dataset(dataset: DatasetDict, dataset_percentage: Optional[float]) -> DatasetDict:
    """
    Filters the dataset based on the given percentage.

    Args:
        dataset: The dataset dictionary.
        dataset_percentage: The percentage of the original dataset to retain.

    Returns:
        The filtered dataset dictionary.
    """
    if dataset_percentage is None:
        raise ValueError("dataset_percentage cannot be None")

    train_new_size = round(len(dataset["train"]) * dataset_percentage / 100)
    val_new_size = round(len(dataset["validation"]) * dataset_percentage / 100)
    dataset["train"] = dataset["train"].select(range(train_new_size))
    dataset["validation"] = dataset["validation"].select(range(val_new_size))
    return dataset


def load_dataset_and_preprocess(config: Dict[str, Any], tokenizer) -> Dataset:
    """
    Loads and preprocesses the dataset according to the given configuration.

    Args:
        config: The configuration object containing dataset and preprocessing parameters.
        tokenizer: The tokenizer to be used for preprocessing the dataset.

    Returns:
        The preprocessed dataset.
    """
    logger.info("Loading dataset...")
    dataset = load_dataset_from_folders(config["dataset_path"])
    for element in dataset:
        logger.info(f"Original size of {element}: {len(dataset[element])}")

    if "dataset_percentage" in config:
        logger.info("Sampling dataset...")
        dataset = filter_dataset(dataset, config.get("dataset_percentage"))
        for element in dataset:
            logger.info(f"Filtered size of {element}: {len(dataset[element])}")

    dataset = preprocess(dataset, tokenizer, config.get("model_max_length", None))

    if "shuffle" in config:
        logger.info("Shuffling dataset...")
        dataset = dataset.shuffle(seed=config.get("dataset_seed", 42))
        logger.info("Done shuffling")

    return dataset
