"""Model Training Script"""

import argparse
import logging
from pathlib import Path

import mlflow  # type: ignore
import torch
import torch.distributed as dist
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer  # type: ignore

from mtl4ad.data_preprocessing import load_dataset_and_preprocess  # type: ignore
from mtl4ad.model import load_model_and_tokenizer_peft  # type: ignore
from mtl4ad.utils import (  # type: ignore
    PeftSavingCallback,
    extract_valid_training_arguments,
    format_name,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()


def initialize_trainer(
    config,
    model,
    tokenizer,
    dataset,
    experiment_name,
    output_dir,
    enable_peft=False,
    peft_config=None,
) -> SFTTrainer:
    """
    Initializes the Trainer for model training.

    Args:
        config: The configuration object containing training parameters.
        model: The model to be trained.
        tokenizer: The tokenizer associated with the model.
        dataset: The dataset to be used for training and evaluation.
        experiment_name: The name of the experiment for logging purposes.
        output_dir: The path where to save the checkpoints.
        enable_peft: Enable perf.

    Returns:
        The initialized Trainer object.
    """
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=experiment_name,
        bf16=IS_BF16_AVAILABLE,
        **config,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[PeftSavingCallback()] if enable_peft else None,
        peft_config=peft_config,
    )


def main() -> None:
    """
    The main function for the training process.
    """
    args = parse_arguments()
    config = load_config(args.config)

    config["dataset_path"] = str(Path(config["dataset_path"]).resolve())
    config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]).resolve())
    if "deepspeed" in config:
        config["deepspeed"] = str(Path(config["deepspeed"]).resolve())

    if "model_name" not in config:
        raise KeyError(
            "The 'model_name' key is missing from the configuration. Terminating script."
        )

    experiment_name = config.get("experiment_name", "train")
    full_experiment_name = format_name(experiment_name)
    output_dir = str(Path(config.get("checkpoint_dir", "tmp")).joinpath(experiment_name))

    enable_peft = config.get("enable_peft", False)
    model, tokenizer, peft_config = load_model_and_tokenizer_peft(config["model_name"], enable_peft)
    dataset = load_dataset_and_preprocess(config, tokenizer)
    config = extract_valid_training_arguments(config)

    trainer = initialize_trainer(
        config, model, tokenizer, dataset, experiment_name, output_dir, enable_peft, peft_config
    )

    is_mlflow_available = config.get("report_to") == "mlflow"

    if is_mlflow_available and (not IS_CUDA_AVAILABLE or torch.distributed.get_rank() == 0):
        mlflow.set_experiment(full_experiment_name)
        mlflow.start_run()
        mlflow.set_tag("mlflow.runName", experiment_name)
        mlflow.pytorch.autolog()
    trainer.train()
    trainer.save_model(str(Path(output_dir).joinpath("last_model")))

    dist.destroy_process_group()

    logger.info("Training completed and model saved.")


if __name__ == "__main__":
    main()
