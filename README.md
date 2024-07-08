# Multi-task LLM for Accelerated Discovery

## Project Overview

This project aims to fine-tune various large language model (LLM) architectures on a multi-task dataset for Accelerated Discovery (AD). It provides scripts for:
- PEFT-based LaMDA fine-tuning of Seq2Seq and CLM models using Hugging Face (HF) transformers Trainer and DeepSpeed
- Inference on the fine-tuned models
- Model evaluation

## Installation

**Standard Installation**

```bash
git clone git@github.com:GT4SD/mtl4ad.git
cd mtl4ad
poetry install
```

**CUDA setup helper**

In the `scripts` directory, there is a script called `torch_setup.sh` that ensures that your PyTorch installation is configured to use the correct CUDA version.

## Data

The multi-task dataset used in this project was described in the paper by Christiofidelis et al. ["Unifying Molecular and Textual Representations via Multi-task Language Modelling"](https://github.com/GT4SD/multitask_text_and_chemistry_t5). The goal is to eventually incorporate additional modalities beyond text and chemistry.

The training dataset should be placed in `src/mtl4ad/resources/train`

**Format and Structure**

The data resides in `PARQUET` format and is automatically loaded using Hugging Face Datasets. Here's an example of a data point:
```
{"source": "Caption the following smile: CC(=O)NC(CC1=CC=C(C=C1)O)C(=O)O", 
"target": "The molecule is an N-acetyl-amino acid that is tyrosine with an amine hydrogen substituted by an acetyl group. It has a role as a human urinary metabolite. It is a tyrosine derivative, a N-acetyl-amino acid and a member of phenols. It derives from a tyrosine."}
```

The data is organized into configurations and splits defined using `YAML` formatting, detailed in a separate `README.md` file within dataset location. Here's a summary of the configurations:

- `main_data`: Training and validation sets as described in Christiofidelis et al. (see above link).
- `spectra_data`: Training and validation sets with an additional spectral modality task (1H-NMR spectra to SMILES).
- `backward_test`, `forward_test`, `d2s_test`, `s2d_test`, and `p2a_test`: Test sets for specific tasks (retrosynthesis, forward synthesis, description to SMILES, SMILES to description, paragraph to actions) described in Christiofidelis et al.
- `spectra2smi_test`: Test set for the 1H-NMR spectra to SMILES task (data curator: Marvin Alberts).

**Sequence Length Considerations**

Based on sequence length analysis, source and target prompts are potentially truncated to a maximum length of 512 (`max_length = 512`) for Seq2Seq modeling. For CLM modeling, both prompts are merged and might be truncated to a maximum length of 1024 (`max_length = 1024`). Remember to implement the correct prompt format (currently supporting mistral-instruct) when using instruction-based models.


## Repository Structure

This repository is organized as follows:
- `scripts`: This directory contains scripts for launching training and inference tasks.
- `src/mtl4ad`: This directory holds the core source code for the projectand a directory called `resources` which contains a sample test dataset and configuration file

## DeepSpeed Integration

This project leverages DeepSpeed through Hugging Face's `accelerate` library for multi-GPU training.
**DeepSpeed Config File**: An example is located at `/configs/deepspeed_config.json`. This file contains DeepSpeed-specific training parameters.

For more details check out [this page](https://huggingface.co/docs/accelerate/usage_guides/deepspeed). 

## Training

The training with `torchrun` command requires specific parameters to configure the training environment. While these can be set directly within the command, it's recommended to use a separate configuration file for improved readability and maintainability.

**An example file is provided below:**
```bash
export CONFIG_FILE="/configs/config_opt_125m.yaml"

./scripts/train.sh ${CONFIG_FILE}  # To initiate training
```

**Explanation:**

- `CONFIG_FILE`: Configuration file for training optimization.


For a list of the arguments please refer to HF TrainingArguments class, as well as custom classes under `src/mtl4ad/train.py`. Further examples of training scripts and launching scripts can be found in this repository under `scripts`.

**Known issues:** 
- Use of Seq2Seq Collator in CLM Modules: To force the learning of eos token addition for CLM models, the Seq2Seq collator is used (cf. [this issue](https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285))
- Model specific prompt formatting in CLM Modules for instruction-based models may need to be implemented.
- For tracking, we employed mlflow throughout the project. However, it seems to only track HF Training Arguments.

## Evaluation

Inference can be launched using the following the same indications as for training but using the `inference.sh` file instead of the `train.sh`. Here is an example:

```bash
export CONFIG_FILE="/configs/config_opt_125m.yaml"

./scripts/inference.sh ${CONFIG_FILE}  # To initiate evaluation
```

## Format and Styling

Use ruff to check the styling and format of your script.

```console
poetry run ruff check .          # Lint all files in the current directory.
poetry run ruff check . --fix    # Lint all files in the current directory, and fix any fixable errors.
poetry run ruff check . --watch  # Lint all files in the current directory, and re-lint on change.
```

**Known issues:** 
- Model specific post-processing may need to be implemented if specific prompt formatting is employed.

## Testing

Use the following command to run the testing
```console
poetry run pytest -sv --cov=mtl4ad tests
```