"""Load a 4-bit LLaVA-compatible VLM and attach QLoRA adapters.

The default model is configurable at the command line.  It is deliberately
kept here rather than being embedded in training code so the verification
pipeline can be reused with a human-approved base model later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


QLORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class QLoRASettings:
    model_id: str
    use_bf16: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def resolve_compute_dtype(prefer_bf16: bool) -> torch.dtype:
    """Use bf16 when supported, otherwise fall back to fp16."""
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# src/models/qlora_vlm.py snippet update

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_quantized_vlm(settings, trainable: bool = True):
    # Use auto class to support Idefics3 / SmolVLM, Llava, and standard VLMs
    processor = AutoProcessor.from_pretrained(settings.model_id)
    
    # Optional 4-bit config check
    bnb_config = None
    if settings.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if settings.use_bf16 else torch.float16,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        settings.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if settings.use_bf16 else torch.float16,
    )

    if trainable:
        model = prepare_model_for_kbit_training(model)
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Adjust target modules per arch if needed
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    compute_dtype = torch.bfloat16 if settings.use_bf16 else torch.float16
    return model, processor, compute_dtype