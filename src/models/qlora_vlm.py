"""Model loading utilities for QLoRA and LoRA fine-tuning of Vision-Language Models (VLMs)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


@dataclass
class QLoRASettings:
    model_id: str
    use_4bit: bool = False
    use_bf16: bool = False


def load_quantized_vlm(settings: QLoRASettings, trainable: bool = True):
    """Loads a vision-language model using AutoModelForVision2Seq with optional 4-bit quantization or 16-bit LoRA."""
    processor = AutoProcessor.from_pretrained(settings.model_id)

    compute_dtype = torch.bfloat16 if settings.use_bf16 else torch.float16

    bnb_config = None
    if settings.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Use AutoModelForVision2Seq to natively support SmolVLM / Idefics3 / Llava architectures
    model = AutoModelForVision2Seq.from_pretrained(
        settings.model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    if trainable:
        if settings.use_4bit:
            model = prepare_model_for_kbit_training(model)

        # Standard linear attention projection targets for LoRA
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
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