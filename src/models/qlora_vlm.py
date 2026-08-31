"""Model loading utilities for QLoRA and LoRA fine-tuning of Vision-Language Models (VLMs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


@dataclass
class QLoRASettings:
    model_id: str
    use_4bit: bool = False
    use_bf16: bool = False


def load_quantized_vlm(
    settings: QLoRASettings,
    trainable: bool = True,
    adapter_path: Optional[Union[str, Path]] = None,
):
    """Loads a vision-language model using AutoModelForImageTextToText."""
    processor = AutoProcessor.from_pretrained(
        adapter_path if adapter_path and Path(adapter_path).exists() else settings.model_id
    )

    compute_dtype = torch.bfloat16 if settings.use_bf16 else torch.float16

    bnb_config = None
    if settings.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Force single CUDA device mapping during training to avoid cross-device tensor errors
    device_map = {"": 0} if trainable else "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        settings.model_id,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map=device_map,
    )

    # 1. Evaluation/Inference with pre-trained adapter
    if adapter_path and Path(adapter_path).exists():
        print(f"[MODEL] Loading trained LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        if not trainable:
            model.eval()
        return model, processor, compute_dtype

    # 2. Fine-tuning setup
    if trainable:
        if settings.use_4bit:
            model = prepare_model_for_kbit_training(model)

        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    return model, processor, compute_dtype