"""Model and Tokenizer Utilities"""

from typing import Any, Optional, Tuple

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from mtl4ad.utils import find_all_linear_names, is_bf16_available  # type: ignore


def load_model_and_tokenizer_peft(
    model_name: str, use_peft: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[Any]]:
    """
    Loads the model and tokenizer.

    Args:
        model_name: The name of the model to load.
        use_peft: Whether to enable PEFT (Lora) for the model.

    Returns:
        A tuple containing the loaded model and tokenizer, and optionally the PEFT config.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if is_bf16_available() else torch.float32
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = None
    if use_peft:
        peft_config = enable_peft(model)

    return model, tokenizer, peft_config


def enable_peft(model) -> torch.nn.Module:
    """
    Enables PEFT (Lora) for the model.

    Args:
        model: The model to enable PEFT on.

    Returns:
        The model with PEFT enabled.
    """
    model.enable_input_require_grads()
    # target_modules = [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    #     "lm_head",
    # ]
    target_modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    return peft_config
    # model = get_peft_model(model, peft_config)
    # return model


def prepare_model_for_training(
    model: torch.nn.Module, tokenizer: AutoTokenizer, train_args
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    Prepares the model for training.

    Args:
        model: The model to prepare.
        tokenizer: The tokenizer to prepare.
        train_args: The training arguments.

    Returns:
        A tuple containing the prepared model and tokenizer.
    """
    max_length = (
        min(tokenizer.model_max_length, train_args.model_max_length)
        if train_args.model_max_length
        else tokenizer.model_max_length
    )
    tokenizer.model_max_length = max_length

    return model, tokenizer
