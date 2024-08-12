"""Testing Script"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from datasets import DatasetDict
from mtl4ad.data_preprocessing import load_dataset_from_folders, filter_dataset  # type: ignore
from mtl4ad.model import load_model_and_tokenizer_peft  # type: ignore
from mtl4ad.train import initialize_trainer  # type: ignore
from mtl4ad.utils import extract_valid_training_arguments, load_config  # type: ignore
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def config_and_tokenizer() -> Tuple[Dict[str, Any], Any, PreTrainedTokenizer, Optional[Any]]:
    """
    Pytest fixture to load configuration and tokenizer for testing.

    Returns:
        A tuple containing the configuration, model, tokenizer objects, and optionally the PEFT config.
    """
    config_path = Path(__file__).resolve().parent.parent / "src/mtl4ad/resources/test/config/test_config.yaml"
    config = load_config(config_path)
    config["dataset_path"] = str(Path(config["dataset_path"]).resolve())
    config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]).resolve())
    enable_peft = config.get("enable_peft", False)
    model, tokenizer, peft_config = load_model_and_tokenizer_peft(config, enable_peft)
    return config, model, tokenizer, peft_config

@pytest.fixture(scope="module")
def dataset(config_and_tokenizer: Tuple[Dict[str, Any], Any, PreTrainedTokenizer, Optional[Any]]) -> DatasetDict:
    """
    Pytest fixture to load, filter, and shuffle the dataset for testing.

    Returns:
        A DatasetDict object ready for training.
    """
    config, _, _, _ = config_and_tokenizer
    dataset = load_dataset_from_folders(config["dataset_path"])

    for element in dataset:
        logger.info(f"Original size of {element}: {len(dataset[element])}")

    if "dataset_percentage" in config:
        logger.info("Sampling dataset...")
        dataset = filter_dataset(dataset, config.get("dataset_percentage"), config.get("n_val_sample"))
        for element in dataset:
            logger.info(f"Filtered size of {element}: {len(dataset[element])}")

    if "shuffle" in config:
        logger.info("Shuffling dataset...")
        dataset = dataset.shuffle(seed=config.get("dataset_seed", 42))
        logger.info("Done shuffling")

    return dataset

def test_training_and_model_saving(config_and_tokenizer: Tuple[Dict[str, Any], Any, PreTrainedTokenizer, Optional[Any]], dataset: DatasetDict):
    """
    Test a single training run and verify model saving.

    Ensures that the model is saved correctly after training.
    """
    config, model, tokenizer, peft_config = config_and_tokenizer
    experiment_name = config.get("experiment_name", "train")
    output_dir = Path(config.get("checkpoint_dir", "tmp")) / experiment_name

    clean_config = extract_valid_training_arguments(config)
    trainer = initialize_trainer(
        clean_config,
        model,
        tokenizer,
        dataset,
        experiment_name,
        str(output_dir),
        config.get("max_seq_length", tokenizer.model_max_length),
        config.get("enable_peft", False),
        peft_config
    )

    trainer.train()
    
    model_save_path = output_dir / "last_model"
    assert model_save_path.exists(), "Model save path should exist after training"
    logger.info("Training run and model saving test completed successfully.")
