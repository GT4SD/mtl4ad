"""Model and Tokenizer Utilities"""

from typing import Any, Optional, Tuple, Dict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from accelerate import PartialState


from mtl4ad.utils import find_all_linear_names, is_bf16_available  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer_peft(
    config: Dict[str, Any], use_peft: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[Any]]:
    """
    Loads the model and tokenizer.

    Args:
        model_name: The name of the model to load.
        use_peft: Whether to enable PEFT (Lora) for the model.

    Returns:
        A tuple containing the loaded model and tokenizer, and optionally the PEFT config.
    """
    model_name=config["model_name"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if is_bf16_available() else torch.float32,
        # use_flash_attention_2=True,
        quantization_config=bnb_config,
        # load_in_4bit=True,
        device_map={"": PartialState().process_index}
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length = (
    min(tokenizer.model_max_length, config["max_seq_length"])
    if "max_seq_length" in config and config["max_seq_length"] is not None
    else tokenizer.model_max_length
)
    tokenizer.model_max_length = max_length

    peft_config = None
    if use_peft:
        model, peft_config = enable_peft(model)
        logger.info("Peft enabled")

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
 
    model = get_peft_model(model, peft_config)
    return model, peft_config


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
