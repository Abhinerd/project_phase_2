"""Compare the same QLoRA VLM on VizWiz with and without YOLO spatial context."""

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
    parser = argparse.ArgumentParser(
        description="Evaluate a saved QLoRA adapter with and without YOLO context."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"]
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Load the base model in 4-bit (use the same setting as training).",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    return parser.parse_args()


class SpatialExtractor:
    """YOLO segmentation-based object, position, shape and relation extractor."""

    def __init__(self, model_name: str = "yolo11n-seg.pt", conf: float = 0.25):
        import torch
        from ultralytics import YOLO

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.conf = conf

    def process(self, pil_image):
        import cv2
        import numpy as np

        self.image = np.asarray(pil_image.convert("RGB"))
        self.height, self.width = self.image.shape[:2]
        predictions = self.model.predict(
            source=self.image,
            conf=self.conf,
            imgsz=640,
            verbose=False,
            device=self.device,
        )
        self.result = predictions[0]
        self.objects = self._extract_objects()
        self._enrich_with_shapes()
        self.relationships = self._extract_relationships()
        return self

    def _extract_objects(self):
        objects = []
        boxes = getattr(self.result, "boxes", None)
        if boxes is None:
            return objects

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu())
            class_id = int(box.cls[0].detach().cpu())
            class_name = self.result.names[class_id]

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            nx, ny = center_x / self.width, center_y / self.height

            horizontal = (
                "left" if nx < 1 / 3 else "right" if nx > 2 / 3 else "center"
            )
            vertical = "top" if ny < 1 / 3 else "bottom" if ny > 2 / 3 else "middle"
            position = (
                horizontal if vertical == "middle" else f"{vertical}-{horizontal}"
            )

            objects.append({
                "id": index + 1,
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "position": position,
                "shape": "unknown",
            })
        return objects

    def _enrich_with_shapes(self):
        import cv2
        import numpy as np

        masks = getattr(self.result, "masks", None)
        if masks is None:
            return

        for index, obj in enumerate(self.objects):
            if index >= len(masks.data):
                continue
            mask = masks.data[index].detach().cpu().numpy().astype(np.uint8)
            mask = cv2.resize(
                mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
            )
            _, _, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / float(height) if height > 0 else 0.0

            if circularity > 0.78:
                shape = "round"
            elif 0.85 <= aspect_ratio <= 1.18:
                shape = "square"
            elif aspect_ratio > 1.35 or aspect_ratio < 0.74:
                shape = "rectangular"
            else:
                shape = "irregular"
            obj["shape"] = shape

    def _extract_relationships(self):
        relationships = []
        for index, first in enumerate(self.objects):
            for second in self.objects[index + 1:]:
                ax, ay = first["center"]
                bx, by = second["center"]
                dx, dy = ax - bx, ay - by
                if abs(dx) > abs(dy):
                    relation = "right of" if dx > 0 else "left of"
                else:
                    relation = "below" if dy > 0 else "above"
                relationships.append(
                    f"{first['class']} (object {first['id']}) is {relation} "
                    f"{second['class']} (object {second['id']})"
                )
        return relationships

    def to_ai_context(self) -> str:
        if not self.objects:
            return ""

        entities = [
            f"{obj['class']} (position={obj['position']}, shape={obj['shape']})"
            for obj in self.objects
        ]
        context = "Detected objects: " + "; ".join(entities) + "."
        if self.relationships:
            context += " Spatial relationships: " + "; ".join(
                self.relationships[:6]
            ) + "."
        return context


def load_image(image_path: Path, allow_missing: bool):
    from PIL import Image

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((448, 448), Image.LANCZOS)
        return image
    except (FileNotFoundError, OSError) as exc:
        if not allow_missing:
            raise FileNotFoundError(f"Cannot load image: {image_path}") from exc
        return Image.new("RGB", (448, 448), color=(0, 0, 0))


def evaluate_model(
    model,
    processor,
    records,
    image_root,
    allow_missing,
    device,
    use_yolo: bool,
    spatial_extractor=None,
):
    """Evaluate one condition. The same model, records and metric are used."""
    import torch
    from src.data.vizwiz import build_conversation
    from src.evaluation import vizwiz_ans

    if use_yolo and spatial_extractor is None:
        raise ValueError("YOLO evaluation requested but no SpatialExtractor was supplied.")

    model.eval()
    results = []
    started = time.perf_counter()

    with torch.inference_mode():
        for item in tqdm(
            records,
            desc="Evaluating with YOLO" if use_yolo else "Evaluating without YOLO",
        ):
            image = load_image(Path(image_root) / item["image"], allow_missing)
            question = item["question"]

            # Only the YOLO condition transforms/enriches the input question.
            if use_yolo:
                spatial_extractor.process(image)
                spatial_context = spatial_extractor.to_ai_context()
                if spatial_context:
                    question = (
                        f"[Visual spatial context: {spatial_context}]\n"
                        f"Question: {question}"
                    )

            conversation = build_conversation(question, target=None)
            prompt = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(device)

            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
            input_length = inputs["input_ids"].shape[-1]
            prediction = processor.decode(
                generated[0][input_length:],
                skip_special_tokens=True,
            ).strip()

            score = vizwiz_ans(prediction, item["answers"])
            results.append({
                "ans": score,
                "prediction": prediction,
                "answer_type": item.get("answer_type", "other"),
                "image": item["image"],
                "question": item["question"],
            })

    return results, time.perf_counter() - started


def main() -> None:
    import torch
    from src.data.vizwiz import prepare_records
    from src.evaluation import compute_all_metrics
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    args = arguments()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Use the requested split (not a hard-coded "test").
    records = prepare_records(
        args.dataset, args.cache_dir, args.max_samples, args.split
    )
    if args.max_samples > 0:
        records = records[:args.max_samples]
    if not records:
        raise ValueError(f"No records found for split '{args.split}'.")

    # Load the base model + adapter exactly once, then evaluate both conditions.
    model, processor, _ = load_quantized_vlm(
        QLoRASettings(
            args.model_id,
            use_4bit=args.use_4bit,
            use_bf16=not args.fp16,
        ),
        adapter_path=str(args.adapter_path),
        trainable=False,
    )
    device = next(model.parameters()).device

    # Condition 1: original image + original question (no YOLO).
    no_yolo_results, no_yolo_elapsed = evaluate_model(
        model=model,
        processor=processor,
        records=records,
        image_root=args.image_root,
        allow_missing=args.allow_missing_images,
        device=device,
        use_yolo=False,
    )
    no_yolo_metrics = compute_all_metrics(no_yolo_results)

    # Condition 2: same image + question enriched with YOLO spatial context.
    # YOLO is initialized only for this condition.
    spatial_extractor = SpatialExtractor(
        model_name=args.yolo_model,
        conf=args.yolo_confidence,
    )
    yolo_results, yolo_elapsed = evaluate_model(
        model=model,
        processor=processor,
        records=records,
        image_root=args.image_root,
        allow_missing=args.allow_missing_images,
        device=device,
        use_yolo=True,
        spatial_extractor=spatial_extractor,
    )
    yolo_metrics = compute_all_metrics(yolo_results)

    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def make_payload(results, metrics, elapsed, condition):
        return {
            "split": args.split,
            "condition": condition,
            "samples": len(results),
            "mean_vizwiz_ans": metrics["overall_ans"],
            "per_class_ans": metrics["per_class_ans"],
            "type_accuracy": metrics["type_accuracy"],
            "type_macro_f1": metrics["type_macro_f1"],
            "type_macro_precision": metrics["type_macro_precision"],
            "type_macro_recall": metrics["type_macro_recall"],
            "per_class_type_metrics": metrics["per_class_type_metrics"],
            "latency_ms_per_query": (
                elapsed / len(results) * 1000 if results else 0.0
            ),
            "elapsed_seconds": elapsed,
            "peak_vram_gib": peak_vram_gib,
            "predictions": results,
        }

    no_yolo_payload = make_payload(
        no_yolo_results, no_yolo_metrics, no_yolo_elapsed, "without_yolo"
    )
    yolo_payload = make_payload(
        yolo_results, yolo_metrics, yolo_elapsed, "with_yolo"
    )

    (args.output_dir / "final_evaluation_without_yolo.json").write_text(
        json.dumps(no_yolo_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "final_evaluation_with_yolo.json").write_text(
        json.dumps(yolo_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison = {
        "split": args.split,
        "samples": len(records),
        "without_yolo": {
            "mean_vizwiz_ans": no_yolo_metrics["overall_ans"],
            "type_accuracy": no_yolo_metrics["type_accuracy"],
            "type_macro_f1": no_yolo_metrics["type_macro_f1"],
            "latency_ms_per_query": no_yolo_payload["latency_ms_per_query"],
        },
        "with_yolo": {
            "mean_vizwiz_ans": yolo_metrics["overall_ans"],
            "type_accuracy": yolo_metrics["type_accuracy"],
            "type_macro_f1": yolo_metrics["type_macro_f1"],
            "latency_ms_per_query": yolo_payload["latency_ms_per_query"],
        },
        "delta_yolo_minus_no_yolo": {
            "mean_vizwiz_ans": (
                yolo_metrics["overall_ans"] - no_yolo_metrics["overall_ans"]
            ),
            "type_accuracy": (
                yolo_metrics["type_accuracy"] - no_yolo_metrics["type_accuracy"]
            ),
            "type_macro_f1": (
                yolo_metrics["type_macro_f1"] - no_yolo_metrics["type_macro_f1"]
            ),
        },
    }
    (args.output_dir / "final_evaluation_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL EVALUATION COMPARISON")
    print(f"Split: {args.split} | Samples per condition: {len(records)}")
    print(
        f"Without YOLO ANS: {no_yolo_metrics['overall_ans']:.4f} | "
        f"Type accuracy: {no_yolo_metrics['type_accuracy']:.4f} | "
        f"Macro F1: {no_yolo_metrics['type_macro_f1']:.4f}"
    )
    print(
        f"With YOLO ANS:    {yolo_metrics['overall_ans']:.4f} | "
        f"Type accuracy: {yolo_metrics['type_accuracy']:.4f} | "
        f"Macro F1: {yolo_metrics['type_macro_f1']:.4f}"
    )
    print(
        f"Latency without YOLO: {no_yolo_payload['latency_ms_per_query']:.2f} ms/query"
    )
    print(
        f"Latency with YOLO:    {yolo_payload['latency_ms_per_query']:.2f} ms/query"
    )
    print(f"Peak VRAM: {peak_vram_gib:.3f} GiB")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
