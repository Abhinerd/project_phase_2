"""Mini generated-answer evaluation for a saved QLoRA smoke-test adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--adapter-path", type=Path, default=ROOT / "artifacts/checkpoints/fast_test_adapter")
    parser.add_argument("--dataset", type=Path, default=ROOT / "phase_1/data/vizwiz_train_hindi.json")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/checkpoints/fast_test_adapter")
    parser.add_argument("--num-train-samples", type=int, default=50, help="Training prefix to skip for the deterministic smoke-test holdout.")
    parser.add_argument("--num-val-samples", type=int, default=20)
    parser.add_argument("--generation-max-new-tokens", type=int, default=24)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_image(path: Path, allow_missing: bool):
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        if allow_missing:
            return Image.new("RGB", (224, 224), color=(0, 0, 0))
        raise FileNotFoundError(f"Cannot load {path}; provide --image-root or use smoke-test fallback.") from exc


def main() -> None:
    args = arguments()
    import torch

    from src.data.vizwiz import prepare_records
    from src.evaluation import vizwiz_ans
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA because the adapter is loaded with 4-bit bitsandbytes.")
    torch.cuda.reset_peak_memory_stats()
    all_records = prepare_records(
        args.dataset,
        args.cache_dir,
        args.num_train_samples + args.num_val_samples,
        "evaluation_source",
    )
    records = all_records[args.num_train_samples :]
    model, processor, _ = load_quantized_vlm(
        QLoRASettings(args.model_id, use_bf16=not args.fp16), adapter_path=str(args.adapter_path), trainable=False
    )
    model.eval()
    device = next(model.parameters()).device
    results = []
    with torch.inference_mode():
        for item in records:
            image = load_image(args.image_root / item["image"], args.allow_missing_images)
            inputs = processor(text=item["generation_prompt"], images=image, return_tensors="pt").to(device)
            generated = model.generate(**inputs, max_new_tokens=args.generation_max_new_tokens, do_sample=False)
            prediction = processor.decode(generated[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            results.append({"index": item["index"], "prediction": prediction, "references": item["answers"], "ans": vizwiz_ans(prediction, item["answers"])})

    mean_ans = sum(row["ans"] for row in results) / len(results) if results else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": len(results),
        "mean_vizwiz_ans": mean_ans,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "predictions": results,
    }
    (args.output_dir / "fast_test_evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FAST TEST EVALUATION COMPLETE")
    print(f"mean_vizwiz_ans={mean_ans:.4f}")
    print(f"elapsed_seconds={payload['elapsed_seconds']}")
    print(f"peak_vram_gib={payload['peak_vram_gib']}")
    print(f"results_path={args.output_dir / 'fast_test_evaluation.json'}")


if __name__ == "__main__":
    main()
