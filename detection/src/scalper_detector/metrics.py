"""Classification metrics and cutoff selection for a separate calibration split."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _finite_array(values: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validated_pair(
    labels: object, values: object, *, name: str, probabilities: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    y = _finite_array(labels, "labels")
    scores = _finite_array(values, name)
    if y.ndim != 1 or scores.ndim != 1 or y.size == 0 or scores.size != y.size:
        raise ValueError(f"labels and {name} must be nonempty one-dimensional arrays of equal length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary values (0 or 1)")
    if probabilities and np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return y, scores


def sigmoid(logits: object) -> np.ndarray:
    """Numerically stable sigmoid, retaining the shape of the input array."""
    values = _finite_array(logits, "logits")
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    stops = np.r_[starts[1:], labels.size]
    positive_counts = np.add.reduceat(labels[order], starts)
    average_ranks = (starts + stops + 1.0) / 2.0
    rank_sum = float(np.dot(positive_counts, average_ranks))
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    stops = np.r_[starts[1:], labels.size]
    positive_counts = np.add.reduceat(labels[order], starts)
    precision = np.cumsum(positive_counts) / stops
    return float(np.dot(positive_counts / positives, precision))


def binary_metrics(
    labels: object, probabilities: object, threshold: float = 0.5
) -> dict[str, object]:
    """Calculate binary metrics with ties grouped in ranking metrics.

    Predictions are positive at ``probability >= threshold``. Undefined
    precision, recall, and F1 are zero. ROC AUC needs both classes; average
    precision needs at least one positive. Undefined ranking metrics are None.
    The returned values can be serialized directly as strict JSON.
    """
    y, scores = _validated_pair(labels, probabilities, name="probabilities", probabilities=True)
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number in [0, 1]") from exc
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be a finite number in [0, 1]")

    predicted = scores >= cutoff
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    clipped = np.clip(scores, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return {
        "n_samples": int(y.size),
        "prevalence": float(y.mean()),
        "accuracy": float((tp + tn) / y.size),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": _roc_auc(y, scores),
        "average_precision": _average_precision(y, scores),
        "brier_score": float(np.mean((scores - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_threshold(
    labels: object, probabilities: object, *, target_precision: float | None = None
) -> float:
    """Select a calibration cutoff over observed probabilities and 0.5.

    With no precision target, maximize F1 as before. With a target, maximize
    recall among cutoffs meeting that precision. If no cutoff meets it, maximize
    precision and then recall. Remaining ties always favor the higher cutoff.
    The caller must report an unmet target explicitly rather than imply it was
    achieved. These are calibration results, not guarantees for future orders.
    """
    if target_precision is not None and (
        isinstance(target_precision, bool)
        or not isinstance(target_precision, Real)
        or not math.isfinite(target_precision)
        or not 0 < target_precision <= 1
    ):
        raise ValueError("target_precision must be a finite number in (0, 1].")
    y, scores = _validated_pair(labels, probabilities, name="probabilities", probabilities=True)
    candidates = np.unique(np.r_[scores, 0.5])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    prefix_positives = np.r_[0.0, np.cumsum(y[order])]
    starts = np.searchsorted(sorted_scores, candidates, side="left")
    positives = float(y.sum())
    true_positives = positives - prefix_positives[starts]
    predicted_positives = y.size - starts
    if target_precision is not None:
        precision = np.divide(true_positives, predicted_positives, out=np.zeros_like(true_positives),
                              where=predicted_positives > 0)
        eligible = precision >= target_precision
        if not eligible.any():
            eligible = precision == precision.max()
        eligible &= true_positives == true_positives[eligible].max()
        return float(candidates[np.flatnonzero(eligible)[-1]])
    denominator = predicted_positives + positives
    f1 = np.divide(2.0 * true_positives, denominator, out=np.zeros_like(true_positives), where=denominator > 0)
    return float(candidates[np.flatnonzero(f1 == f1.max())[-1]])


def fit_temperature(logits: object, labels: object) -> float:
    """Fit positive scalar temperature on calibration data by minimizing BCE.

    A deterministic bounded golden-section search uses log-temperature in
    [-3, 3]. Including T=1 guarantees the fitted objective is no worse than the
    uncalibrated calibration objective. Calibration must not use test labels.
    """
    y, values = _validated_pair(labels, logits, name="logits")

    def loss(log_temperature: float) -> float:
        scaled = values * math.exp(-log_temperature)
        # A correct high-magnitude logit contributes exactly zero, avoiding
        # cancellation between max(logit, 0) and label * logit.
        signed = (1.0 - 2.0 * y) * scaled
        return float(np.mean(np.logaddexp(0.0, signed)))

    lower, upper = -3.0, 3.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss, right_loss = loss(left), loss(right)
    for _ in range(80):
        if left_loss < right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - ratio * (upper - lower)
            left_loss = loss(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + ratio * (upper - lower)
            right_loss = loss(right)

    candidates = [0.0, -3.0, 3.0, (lower + upper) / 2.0]
    best = min(candidates, key=loss)
    return float(math.exp(best))
