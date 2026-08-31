"""Fast end-to-end QLoRA smoke training for the available VizWiz-Hindi data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"  # existing Phase 1 notebook model; configurable, not a Phase 2 decision


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


def move_to_model_device(batch, model):
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in batch.items()}


def main() -> None:
    args = arguments()
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.data.vizwiz import LlavaDataCollator, VizWizHindiDataset, cache_tokenized_text, prepare_records
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit QLoRA verification pipeline requires a CUDA-capable GPU.")
    torch.cuda.reset_peak_memory_stats()

    train_records = prepare_records(args.dataset, args.cache_dir, args.num_train_samples, "train")
    # The source artifact has no held-out Hindi file; reserve the next records only for this smoke test.
    all_records = prepare_records(args.dataset, args.cache_dir, args.num_train_samples + args.num_val_samples, "train_plus_val")
    val_records = all_records[args.num_train_samples :]
    model, processor, compute_dtype = load_quantized_vlm(
        QLoRASettings(model_id=args.model_id, use_bf16=not args.fp16), trainable=True
    )
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
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for _epoch in range(args.num_train_epochs):
        for batch_index, batch in enumerate(loader):
            batch = move_to_model_device(batch, model)
            with torch.autocast("cuda", dtype=compute_dtype):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(loss.item() * args.gradient_accumulation_steps)
                print(f"train_step={step} loss={losses[-1]:.4f}")
                if step >= args.max_train_steps:
                    break
        if step >= args.max_train_steps:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    peak_vram_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    metrics = {
        "mode": "fast_test",
        "model_id": args.model_id,
        "train_samples": len(train_records),
        "validation_samples_reserved": len(val_records),
        "epochs_requested": args.num_train_epochs,
        "optimizer_steps": step,
        "losses": losses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": peak_vram_gib,
        "vram_limit_gib": args.max_vram_gib,
        "vram_within_limit": peak_vram_gib <= args.max_vram_gib,
        "compute_dtype": str(compute_dtype),
        "missing_images_allowed": args.allow_missing_images,
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"adapter_path={args.output_dir}")
    print(f"elapsed_seconds={metrics['elapsed_seconds']}")
    print(f"peak_vram_gib={metrics['peak_vram_gib']}")
    if not metrics["vram_within_limit"]:
        raise RuntimeError(
            f"Peak VRAM {peak_vram_gib:.3f} GiB exceeds the configured "
            f"{args.max_vram_gib:.3f} GiB safety limit."
        )
    print("FAST TEST COMPLETE")
    print("Run scripts/evaluate.py with the same --model-id and --adapter-path to verify generation.")


if __name__ == "__main__":
    main()
