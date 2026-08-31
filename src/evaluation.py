"""Small, dependency-free VizWiz ANS evaluator."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\"'")


def vizwiz_ans(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    matches = sum(normalized_prediction == normalize_answer(answer) for answer in list(answers)[:10])
    return min(1.0, matches / 3.0)
