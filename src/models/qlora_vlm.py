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