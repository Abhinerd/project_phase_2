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


def load_quantized_vlm(
    settings: QLoRASettings,
    *,
    adapter_path: str | None = None,
    trainable: bool = True,
):
    """Return ``(model, processor, compute_dtype)`` with memory safeguards enabled."""
    compute_dtype = resolve_compute_dtype(settings.use_bf16)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        settings.model_id,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(settings.model_id)
    # KV cache conflicts with gradient checkpointing during training, but is useful
    # for the short generation pass in evaluation.
    model.config.use_cache = not trainable

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QLORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora)

    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, processor, compute_dtype
