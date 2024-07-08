"""Testing Script"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from datasets import Dataset
from mtl4ad.data_preprocessing import load_dataset_from_folders, preprocess  # type: ignore
from mtl4ad.model import load_model_and_tokenizer_peft  # type: ignore
from mtl4ad.train import initialize_trainer  # type: ignore
from mtl4ad.utils import extract_valid_training_arguments, load_config  # type: ignore
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()


def load_dataset_and_preprocess(config: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Loads and preprocesses the dataset according to the given configuration.

    Args:
        config: The configuration object containing dataset and preprocessing parameters.
        tokenizer: The tokenizer to be used for preprocessing the dataset.

    Returns:
        The preprocessed dataset.
    """
    logger.info("Loading dataset...")
    dataset_path = (
        Path(__file__).resolve().parent.parent / "src/mtl4ad" / config.get("dataset_path", None)
    )
    dataset = load_dataset_from_folders(dataset_path)
    logger.info(f"Original dataset size: {len(dataset)}")

    dataset = preprocess(dataset, tokenizer, config.get("model_max_length", None))
    return dataset


@pytest.fixture
def config_and_tokenizer() -> Tuple[Dict[str, Any], Any, PreTrainedTokenizer, Optional[Any]]:
    """
    Pytest fixture to load configuration and tokenizer for testing.

    Returns:
        A tuple containing the configuration, model, tokenizer objects, and optionally the PEFT config.
    """
    config_path = (
        Path(__file__).resolve().parent.parent
        / "src/mtl4ad/resources/test/config"
        / "test_config.yaml"
    )
    config = load_config(config_path)
    config["dataset_path"] = str(Path(config["dataset_path"]).resolve())
    config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]).resolve())
    enable_peft = config.get("enable_peft", False)
    model, tokenizer, peft_config = load_model_and_tokenizer_peft(config["model_name"], enable_peft)
    return config, model, tokenizer, peft_config


def test_training_initialization(
    config_and_tokenizer: Tuple[Dict[str, Any], Any, PreTrainedTokenizer, Optional[Any]],
):
    """
    Test the training setup and initialization.
    """
    config, model, tokenizer, peft_config = config_and_tokenizer
    experiment_name = config.get("experiment_name", "train")
    enable_peft = config.get("enable_peft", False)
    dataset = load_dataset_and_preprocess(config, tokenizer)

    output_dir = str(Path(config.get("checkpoint_dir", "tmp")).joinpath(experiment_name))

    config = extract_valid_training_arguments(config)
    trainer = initialize_trainer(
        config, model, tokenizer, dataset, experiment_name, output_dir, enable_peft, peft_config
    )

    trainer.train()
    logger.info("Test completed successfully.")
