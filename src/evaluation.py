"""Small, dependency-free VizWiz ANS evaluator with Type metrics."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
except ImportError:
    raise ImportError("scikit-learn is required for type metrics. `pip install scikit-learn`")


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\"'")


def vizwiz_ans(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    matches = sum(normalized_prediction == normalize_answer(answer) for answer in list(answers)[:10])
    return min(1.0, matches / 3.0)


def infer_answer_type(prediction: str) -> str:
    """Heuristically map a generated answer to a VizWiz answer_type."""
    pred_clean = normalize_answer(prediction)
    if not pred_clean:
        return "unanswerable"
    
    # English + Hindi Yes/No
    if pred_clean in ["yes", "no", "हां", "हाँ", "नहीं"]:
        return "yes/no"
    
    # English + Hindi Unanswerable
    if pred_clean in ["unanswerable", "unanswerable.", "unanswerable?", "बिना-जवाब", "जवाब नहीं", "उत्तर नहीं"]:
        return "unanswerable"
        
    # English + Hindi Numbers (0-9 and ०-९)
    if re.match(r"^[0-9०-९]+$", pred_clean):
        return "number"
        
    return "other"


def compute_all_metrics(results: list[dict]) -> dict:
    """Computes ANS, Type Accuracy, P/R/F1, and Latency."""
    # ANS Metrics
    mean_ans = sum(r["ans"] for r in results) / len(results) if results else 0.0
    
    class_ans_scores = defaultdict(list)
    for r in results:
        class_ans_scores[r["answer_type"]].append(r["ans"])
    
    per_class_ans = {k: sum(v)/len(v) for k, v in class_ans_scores.items()}

    # Type Classification Metrics
    y_true = [r["answer_type"] for r in results]
    y_pred = [infer_answer_type(r["prediction"]) for r in results]
    
    labels = sorted(list(set(y_true + y_pred)))
    
    type_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    
    # Handle cases where a label might not be predicted at all
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    
    per_class_prf = {}
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for i, label in enumerate(labels):
        per_class_prf[label] = {
            "precision": p_per[i],
            "recall": r_per[i],
            "f1": f1_per[i]
        }

    return {
        "overall_ans": mean_ans,
        "per_class_ans": per_class_ans,
        "type_accuracy": type_acc,
        "type_macro_precision": precision,
        "type_macro_recall": recall,
        "type_macro_f1": f1,
        "per_class_type_metrics": per_class_prf
    }
