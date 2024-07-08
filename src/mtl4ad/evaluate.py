"""Model Evaluation Script"""

import argparse

from transformers import Trainer, TrainingArguments

from mtl4ad.datasets import load_dataset  # type: ignore
from mtl4ad.model import load_model  # type: ignore
from mtl4ad.utils import load_config  # type: ignore


def main() -> None:
    """
    Main function to run the evaluation script. It parses the configuration file,
    loads the model and dataset, and runs evaluation using the Trainer API.
    """
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model_name"]
    max_length = config["model_max_length"]

    model, tokenizer = load_model(model_name, max_length)

    dataset_path = config["dataset_path"]
    dataset = load_dataset(path=dataset_path)["validation"]

    eval_args = TrainingArguments(
        per_device_eval_batch_size=config["batch_size"],
        output_dir=config["checkpoint_dir"],
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=eval_args,
        eval_dataset=dataset,
    )

    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()
