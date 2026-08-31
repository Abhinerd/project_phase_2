"""Verbose end-to-end QLoRA smoke training for VizWiz-Hindi data with extended logging and checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=ROOT / "phase_1/data/vizwiz_train_hindi.json")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/checkpoints/fast_test_adapter")
    parser.add_argument("--num-train-samples", type=int, default=50)
    parser.add_argument("--num-val-samples", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-vram-gib", type=float, default=16.0)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Prefer fp16 even where bf16 is available.")
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n{'=' * 65}\n{title.upper()}\n{'=' * 65}")


def move_to_model_device(batch, model):
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in batch.items()}


def main() -> None:
    args = arguments()
    
    print_section("Pre-Flight Checks & Configuration")
    print(f"[CHECK] Python Version     : {sys.version.split()[0]}")
    print(f"[CHECK] Root Directory     : {ROOT}")
    print(f"[CHECK] Dataset Path       : {args.dataset} (Exists: {args.dataset.exists()})")
    print(f"[CHECK] Image Root Path    : {args.image-root} (Exists: {args.image-root.exists()})")
    print(f"[CHECK] Target Model ID    : {args.model_id}")
    print(f"[CHECK] Output Directory   : {args.output_dir}")
    
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found at: {args.dataset}")

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.data.vizwiz import LlavaDataCollator, VizWizHindiDataset, cache_tokenized_text, prepare_records
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    print(f"[CHECK] PyTorch Version    : {torch.__version__}")
    print(f"[CHECK] CUDA Available     : {torch.cuda.is_available()}")

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit QLoRA verification pipeline requires a CUDA-capable GPU.")
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[CHECK] GPU Device Name    : {gpu_name}")
    print(f"[CHECK] GPU VRAM Total     : {total_vram_gib:.2f} GiB")
    
    torch.cuda.reset_peak_memory_stats()

    print_section("Data Preparation & Tokenization")
    print(f"[DATA] Loading train records (N={args.num_train_samples})...")
    train_records = prepare_records(args.dataset, args.cache_dir, args.num_train_samples, "train")
    
    print(f"[DATA] Reserving validation records (N={args.num_val_samples})...")
    all_records = prepare_records(args.dataset, args.cache_dir, args.num_train_samples + args.num_val_samples, "train_plus_val")
    val_records = all_records[args.num_train_samples :]

    print_section("Loading Quantized Model (4-bit QLoRA)")
    load_start = time.perf_counter()
    model, processor, compute_dtype = load_quantized_vlm(
        QLoRASettings(model_id=args.model_id, use_bf16=not args.fp16), trainable=True
    )
    print(f"[MODEL] Loaded in {time.perf_counter() - load_start:.2f}s")
    print(f"[MODEL] Compute Data Type : {compute_dtype}")

    # Parameter Stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable Params  : {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    print("[DATA] Caching tokenized text...")
    cache_tokenized_text(train_records, processor, args.cache_dir, "fast_train")
    cache_tokenized_text(val_records, processor, args.cache_dir, "fast_val")

    train_set = VizWizHindiDataset(train_records, args.image_root, args.allow_missing_images)
    loader = DataLoader(
        train_set,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=LlavaDataCollator(processor),
    )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    
    print_section("Training Loop Started")
    print(f"[TRAIN] Batch Size        : {args.per_device_train_batch_size}")
    print(f"[TRAIN] Grad Accum Steps  : {args.gradient_accumulation_steps}")
    print(f"[TRAIN] Effective Batch   : {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"[TRAIN] Target Max Steps   : {args.max_train_steps}\n")

    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    step_start_time = time.perf_counter()

    for epoch in range(args.num_train_epochs):
        print(f"--- Epoch {epoch + 1}/{args.num_train_epochs} ---")
        for batch_index, batch in enumerate(loader):
            batch = move_to_model_device(batch, model)
            
            with torch.autocast("cuda", dtype=compute_dtype):
                loss_out = model(**batch).loss
                loss = loss_out / args.gradient_accumulation_steps
            
            loss.backward()

            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                
                step_loss = loss.item() * args.gradient_accumulation_steps
                losses.append(step_loss)
                
                step_duration = time.perf_counter() - step_start_time
                current_vram = torch.cuda.memory_allocated() / (1024**3)
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                
                print(
                    f"[STEP {step:02d}/{args.max_train_steps}] "
                    f"Loss: {step_loss:.4f} | "
                    f"Time: {step_duration:.2f}s | "
                    f"VRAM Allocated: {current_vram:.2f} GiB | "
                    f"VRAM Peak: {peak_vram:.2f} GiB"
                )
                
                step_start_time = time.perf_counter()
                
                if step >= args.max_train_steps:
                    print(f"[TRAIN] Reached max requested steps ({args.max_train_steps}). Halting.")
                    break
        if step >= args.max_train_steps:
            break

    print_section("Saving Artifacts & Metrics")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SAVE] Saving adapter weights to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    peak_vram_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    total_elapsed = round(time.perf_counter() - started, 3)

    metrics = {
        "mode": "fast_test_verbose",
        "model_id": args.model_id,
        "train_samples": len(train_records),
        "validation_samples_reserved": len(val_records),
        "epochs_requested": args.num_train_epochs,
        "optimizer_steps": step,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "elapsed_seconds": total_elapsed,
        "peak_vram_gib": peak_vram_gib,
        "vram_limit_gib": args.max_vram_gib,
        "vram_within_limit": peak_vram_gib <= args.max_vram_gib,
        "compute_dtype": str(compute_dtype),
        "missing_images_allowed": args.allow_missing_images,
    }

    metrics_file = args.output_dir / "training_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[SAVE] Saved training metrics to: {metrics_file}")

    print_section("Execution Summary")
    print(f"  * Adapter Path       : {args.output_dir}")
    print(f"  * Total Time Elapsed : {total_elapsed} s")
    print(f"  * Peak VRAM Used     : {peak_vram_gib} GiB / Limit {args.max_vram_gib} GiB")
    print(f"  * Steps Completed    : {step}")
    if losses:
        print(f"  * Initial Loss       : {losses[0]:.4f}")
        print(f"  * Final Loss         : {losses[-1]:.4f}")

    if not metrics["vram_within_limit"]:
        raise RuntimeError(
            f"Peak VRAM {peak_vram_gib:.3f} GiB exceeds the configured "
            f"{args.max_vram_gib:.3f} GiB safety limit."
        )

    print_section("Fast Smoke Test Complete")
    print("Run `python scripts/evaluate.py --model-id ... --adapter-path ...` to verify generation.")


if __name__ == "__main__":
    main()