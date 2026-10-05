from __future__ import annotations

import math
import re
import string
from collections import Counter
from statistics import mean
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else float("nan")


def best_exact_match(prediction: str, references: Sequence[str]) -> float:
    return max((exact_match(prediction, reference) for reference in references), default=0.0)


def best_token_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, reference) for reference in references), default=0.0)


def best_rouge_l(prediction: str, references: Sequence[str]) -> float:
    return max((rouge_l(prediction, reference) for reference in references), default=0.0)


def ruler_score(prediction: str, references: Sequence[str], task: str) -> float:
    """Official RULER-style case-insensitive substring score.

    QA tasks pass when any accepted answer appears. Retrieval and aggregation
    tasks receive fractional credit for every requested answer that appears.
    """

    if not references:
        return 0.0
    normalized_prediction = prediction.lower()
    hits = [float(reference.lower() in normalized_prediction) for reference in references]
    return max(hits) if task.split("_", maxsplit=1)[0] == "qa" else sum(hits) / len(hits)


def quality_score(prediction: str, reference: str) -> float:
    """Score a generated QuALITY answer as an A/B/C/D multiple-choice label."""

    text = prediction.strip()
    explicit = re.search(
        r"(?:answer|option|choice)(?:\s+is)?\s*[:\-]?\s*[\(\[]?([A-D1-4])\b",
        text,
        flags=re.IGNORECASE,
    )
    leading = re.match(r"^[\s\(\[]*([A-D1-4])(?:\b|[\)\].,:;])", text)
    match = explicit or leading
    if match is None:
        return 0.0
    label = match.group(1).upper()
    if label in "1234":
        label = "ABCD"[int(label) - 1]
    return float(label == reference.upper())
