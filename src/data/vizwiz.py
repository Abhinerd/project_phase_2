"""Portable VizWiz-Hindi records, disk cache, and LLaVA data collation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset


def _cache_key(dataset_path: Path, limit: int, split: str) -> str:
    stat = dataset_path.stat()
    raw = f"{dataset_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{limit}:{split}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def choose_target(record: dict[str, Any]) -> tuple[str, list[str]]:
    """Use a Hindi answer when supplied; otherwise use the first English answer."""
    hindi = record.get("answers_hi") or []
    source = hindi if hindi else record.get("answers") or []
    answers = [entry.get("answer", "").strip() for entry in source if entry.get("answer", "").strip()]
    if not answers:
        answers = ["unanswerable"]
    return answers[0], answers


def format_prompt(question: str, target: str | None = None) -> str:
    prompt = (
        "USER: <image>\n"
        f"{question.strip()}\n"
        "Give a short, direct answer. ASSISTANT:"
    )
    return f"{prompt} {target}" if target is not None else prompt


def prepare_records(dataset_path: Path, cache_dir: Path, limit: int, split: str) -> list[dict[str, Any]]:
    """Create/reuse serializable records and lightweight tokenization cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{_cache_key(dataset_path, limit, split)}"
    prepared_path = cache_dir / f"{stem}_prepared.json"
    if prepared_path.exists():
        return json.loads(prepared_path.read_text(encoding="utf-8"))

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))[:limit]
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        target, answers = choose_target(item)
        question = item.get("question_hi") or item.get("question") or ""
        records.append(
            {
                "index": index,
                "image": item["image"],
                "question": question,
                "target": target,
                "answers": answers,
                "answer_type": item.get("answer_type"),
                "train_prompt": format_prompt(question, target),
                "generation_prompt": format_prompt(question),
            }
        )
    prepared_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def cache_tokenized_text(records: Iterable[dict[str, Any]], processor: Any, cache_dir: Path, name: str) -> Path:
    """Persist tokenizer-only representations for fast audit/repeated preparation.

    The collator still tokenizes multimodal batches because image placeholder expansion
    is model-specific; this cache verifies and preserves the text-tokenization stage.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_tokenized_text.json"
    if path.exists():
        return path
    payload = []
    for item in records:
        encoded = processor.tokenizer(item["train_prompt"], add_special_tokens=True)
        payload.append({"index": item["index"], "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class VizWizHindiDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_root: Path, allow_missing_images: bool = False):
        self.records = records
        self.image_root = image_root
        self.allow_missing_images = allow_missing_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        path = self.image_root / item["image"]
        try:
            image = Image.open(path).convert("RGB")
            image_is_placeholder = False
        except (FileNotFoundError, OSError) as exc:
            if not self.allow_missing_images:
                raise FileNotFoundError(
                    f"Cannot load {path}. Supply --image-root containing VizWiz images, "
                    "or use --allow-missing-images only for a wiring smoke test."
                ) from exc
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            image_is_placeholder = True
        return {**item, "image_data": image, "image_is_placeholder": image_is_placeholder}


class LlavaDataCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.processor(
            text=[item["train_prompt"] for item in examples],
            images=[item["image_data"] for item in examples],
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Supervise only the assistant-answer suffix, leaving prompt tokens masked.
        for row, item in enumerate(examples):
            prefix = self.processor.tokenizer(item["generation_prompt"], add_special_tokens=True)["input_ids"]
            labels[row, : len(prefix)] = -100
        batch["labels"] = labels
        return batch
