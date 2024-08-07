"""Model Training Script"""

import argparse
import logging
from pathlib import Path

import mlflow  # type: ignore
import torch
import torch.distributed as dist
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTConfig, SFTTrainer  # type: ignore

from mtl4ad.data_preprocessing import load_dataset_from_folders, filter_dataset, generate_formatted_prompts  # type: ignore
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
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()

def setup_config(config_path: str) -> dict:
    config = load_config(config_path)
    config["dataset_path"] = str(Path(config["dataset_path"]).resolve())
    config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]).resolve())
    config["deepspeed"] = str(Path(config["deepspeed"]).resolve()) if "deepspeed" in config else None
    if "model_name" not in config:
        raise KeyError("The 'model_name' key is missing from the configuration. Terminating script.")
    return config

def initialize_distributed():
    if IS_CUDA_AVAILABLE and torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

def load_and_prepare_datasets(config: dict) -> dict:
    logger.info("Loading dataset...")
    dataset = load_dataset_from_folders(config["dataset_path"])
    log_dataset_sizes("Original size", dataset)

    if config.get("dataset_percentage"):
        logger.info("Sampling dataset...")
        dataset = filter_dataset(dataset, config.get("dataset_percentage"))
        log_dataset_sizes("Filtered size", dataset)

    if config.get("shuffle"):
        logger.info("Shuffling dataset...")
        dataset = dataset.shuffle(seed=config.get("dataset_seed", 42))
        logger.info("Done shuffling")

    return dataset

def log_dataset_sizes(prefix: str, dataset: dict):
    for element in dataset:
        logger.info(f"{prefix} of {element}: {len(dataset[element])}")

def initialize_trainer(config: dict, dataset: dict, model, tokenizer, collator) -> SFTTrainer:
    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],
        run_name=config["experiment_name"],
        bf16=IS_BF16_AVAILABLE,
        ddp_find_unused_parameters=True,
        remove_unused_columns=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **extract_valid_training_arguments(config),
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[PeftSavingCallback()] if config.get("enable_peft") else None,
        peft_config=config.get("peft_config"),
        formatting_func=generate_formatted_prompts,
    )

def setup_mlflow(config: dict, experiment_name: str):
    if config.get("report_to") == "mlflow":
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.set_tag("mlflow.runName", experiment_name)
        mlflow.pytorch.autolog()

def train_and_save_model(trainer: SFTTrainer, output_dir: str, resume_checkpoint: str):
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_model(str(Path(output_dir).joinpath("last_model")))

def main() -> None:
    args = parse_arguments()
    config = setup_config(args.config)
    experiment_name = format_name(config.get("experiment_name", "train"))

    initialize_distributed()
    
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if local_rank == 0:
        dataset = load_and_prepare_datasets(config)
        if dist.is_initialized():
            torch.distributed.barrier()
    else:
        if dist.is_initialized():
            torch.distributed.barrier()
        dataset = load_dataset_from_folders(config["dataset_path"])

    model, tokenizer, peft_config = load_model_and_tokenizer_peft(config["model_name"], config.get("enable_peft"))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = initialize_trainer(config, dataset, model, tokenizer, collator)

    if not IS_CUDA_AVAILABLE or local_rank == 0:
        setup_mlflow(config, experiment_name)

    train_and_save_model(trainer, config["checkpoint_dir"], config.get("resume_from_checkpoint"))

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("Training completed and model saved.")

if __name__ == "__main__":
    main()