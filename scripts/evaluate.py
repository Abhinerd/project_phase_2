"""Final Evaluation for a saved QLoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--use-4bit", action="store_true", help="Load base model in 4-bit (must match training).")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def evaluate_model(model, processor, records, image_root, allow_missing, device):
    """Exact same loop used in validation to ensure 1:1 parity."""
    import torch
    from PIL import Image
    from src.data.vizwiz import build_conversation
    from src.evaluation import vizwiz_ans
    
    model.eval()
    results = []
    
    with torch.inference_mode():
        for item in tqdm(records, desc="Evaluating"):
            # 1. Image loading
            img_path = Path(image_root) / item["image"]
            try:
                image = Image.open(img_path).convert("RGB")
                image.thumbnail((448, 448), Image.LANCZOS) 
            except (FileNotFoundError, OSError) as exc:
                if not allow_missing:
                    raise FileNotFoundError(f"Cannot load {img_path}") from exc
                image = Image.new("RGB", (448, 448), color=(0, 0, 0))
                
            # 2. Generation logic
            conv = build_conversation(item["question"], target=None)
            prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            
            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            input_length = inputs["input_ids"].shape[-1]
            pred = processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()
            
            # 3. Metric calculation
            ans = vizwiz_ans(pred, item["answers"])
            results.append({"ans": ans, "prediction": pred, "answer_type": item.get("answer_type", "other")})
            
    return results


def main() -> None:
    args = arguments()
    import torch
    from src.data.vizwiz import prepare_records
    from src.evaluation import compute_all_metrics
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    # Load split-aware records
    records = prepare_records(args.dataset, args.cache_dir, args.max_samples, "test")
    
    if args.max_samples > 0:
        records = records[: args.max_samples]

    if not records:
        raise ValueError(f"No records found for split '{args.split}'.")

    model, processor, _ = load_quantized_vlm(
        QLoRASettings(args.model_id, use_4bit=args.use_4bit, use_bf16=not args.fp16),
        adapter_path=str(args.adapter_path),
        trainable=False
    )
    
    device = next(model.parameters()).device
    
    # Call the unified evaluation function
    results = evaluate_model(model, processor, records, args.image_root, args.allow_missing_images, device)

    # Compute all metrics
    metrics = compute_all_metrics(results)
    elapsed = time.perf_counter() - started
    latency_per_query_ms = (elapsed / len(results)) * 1000 if results else 0
    peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "samples": len(results),
        "mean_vizwiz_ans": metrics["overall_ans"],
        "per_class_ans": metrics["per_class_ans"],
        "type_accuracy": metrics["type_accuracy"],
        "type_macro_f1": metrics["type_macro_f1"],
        "type_macro_precision": metrics["type_macro_precision"],
        "type_macro_recall": metrics["type_macro_recall"],
        "per_class_type_metrics": metrics["per_class_type_metrics"],
        "latency_ms_per_query": round(latency_per_query_ms, 2),
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram_gib": round(peak_vram_gib, 3),
        "predictions": results,
    }

    eval_file = args.output_dir / "final_evaluation.json"
    eval_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEVALUATION COMPLETE ({args.split})")
    print(f"Samples: {len(results)}")
    print(f"Overall ANS: {metrics['overall_ans']:.4f}")
    print(f"Type Accuracy: {metrics['type_accuracy']:.4f}")
    print(f"Macro F1: {metrics['type_macro_f1']:.4f}")
    print(f"Latency (ms/query): {latency_per_query_ms:.2f}")
    print(f"Peak VRAM (GiB): {peak_vram_gib:.3f}")

if __name__ == "__main__":
    main()
