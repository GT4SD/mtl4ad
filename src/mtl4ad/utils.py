"""Utility module for configuring, initializing, and managing mtl4ad"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import mlflow  # type: ignore
import torch
import torch.distributed as dist
import yaml  # type: ignore
from transformers import TrainerCallback, TrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parameter_parser() -> argparse.ArgumentParser:
    """
    Builds an ArgumentParser for command-line arguments.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description="Training and Evaluation Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file (YAML or JSON)."
    )
    return parser


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads the configuration from a YAML or JSON file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        The loaded configuration.

    Raises:
        If the config file format is not supported.
    """
    config_path = Path(config_file)
    with config_path.open("r") as file:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(file)
        elif config_path.suffix == ".json":
            return json.load(file)
        else:
            raise ValueError("Unsupported config file format. Use .yaml or .json.")


def is_cuda_available() -> bool:
    """
    Checks if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def is_bf16_available() -> bool:
    """
    Checks if BF16 is available.

    Returns:
        True if BF16 is available, False otherwise.
    """
    return is_cuda_available() and torch.cuda.is_bf16_supported()


def set_environment_variables() -> None:
    """
    Sets environment variables for CUDA.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()


def prepare_output_directory(output_dir: str) -> None:
    """
    Prepares the output directory.

    Args:
        output_dir: Path to the output directory.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)


def log_experiment_details(experiment_name: str, run_name: str) -> None:
    """
    Logs experiment details to MLflow.

    Args:
        experiment_name: Name of the experiment.
        run_name: Name of the run.
    """
    mlflow.set_experiment(experiment_name)
    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.pytorch.autolog()


class PeftSavingCallback(TrainerCallback):
    """
    A callback to save the Lora weights during training.
    """

    def on_save(self, args, state, control, **kwargs) -> None:
        """
        Called on save to store the model weights.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Additional keyword arguments.
        """
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        kwargs["model"].save_pretrained(checkpoint_path)

        for file in checkpoint_path.iterdir():
            if file.name == "pytorch_model.bin":
                file.unlink()

        return control


def find_all_linear_names(model):
    """Get the name of the linear models."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def format_name(name: str) -> str:
    """
    Formats the name with date and job ID.

    Args:
        name: The base name.

    Returns:
        The formatted name.
    """
    job_id = os.getenv("LSB_JOBID")
    current_date = datetime.now().date()
    formatted_date = current_date.strftime("%Y%m%d")
    if job_id is not None:
        formatted_name = f"{formatted_date}_ccc_jobid_{job_id}_{name}"
    else:
        formatted_name = f"{formatted_date}_local_{name}"
    return formatted_name


def setup_distributed_training(num_gpus: int) -> int:
    """
    Sets up distributed training if multiple GPUs are available.

    Args:
        num_gpus: The number of available GPUs.

    Returns:
        The local rank of the process.
    """
    if num_gpus > 0:
        if num_gpus == 1:
            local_rank = 0
            torch.cuda.set_device(local_rank)
            logger.info(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        else:
            dist.init_process_group(backend="nccl")
            local_rank = dist.get_rank()
            torch.cuda.set_device(local_rank)
            logger.info(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    else:
        local_rank = 0
        logger.info("No GPU available.")
    return local_rank


def extract_valid_training_arguments(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the Config instance to a dictionary compatible with TrainingArguments.
    """
    valid_args = set(TrainingArguments.__dataclass_fields__.keys())
    return {k: str(v) if isinstance(v, Path) else v for k, v in config.items() if k in valid_args}
