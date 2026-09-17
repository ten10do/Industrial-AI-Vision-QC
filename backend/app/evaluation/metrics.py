"""Confusion matrix, classification metrics and latency statistics.

Pure arithmetic over :class:`~app.evaluation.records.PredictionRecord`. No
numpy: the populations are O(10^4) at most, and an evaluation that a reviewer
can recompute by hand is worth more here than one that is 100x faster.

Undefined metrics are returned as ``None``, never as 0.0 and never as NaN.
A gate that reads ``None`` must fail closed, and a dashboard that reads
``None`` must render "not defined" rather than a comforting zero. The one
exception is F1 when precision and recall are both defined but zero: that is
a genuine 0.0, and it matches the convention already used by
``steel_patchcore.aggregation.operating_point``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .records import (
    GROUND_TRUTH_NEGATIVE,
    GROUND_TRUTH_POSITIVE,
    PredictionRecord,
    validate_records,
)


def _safe_div(numerator: float, denominator: float) -> float | None:
    """None when the denominator is zero: an undefined rate is not zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, matching ``numpy.percentile(method='linear')``."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    position = (q / 100.0) * (n - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def actual_negative(self) -> int:
        return self.fp + self.tn

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "actual_positive": self.actual_positive,
            "actual_negative": self.actual_negative,
            "predicted_positive": self.predicted_positive,
            # rows = ground truth (negative, positive), cols = prediction
            "matrix": [[self.tn, self.fp], [self.fn, self.tp]],
        }


def confusion_matrix(records: Iterable[PredictionRecord]) -> ConfusionMatrix:
    tp = fp = tn = fn = 0
    for record in records:
        outcome = record.error_type
        if outcome == "TP":
            tp += 1
        elif outcome == "FP":
            fp += 1
        elif outcome == "TN":
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def binary_metrics(matrix: ConfusionMatrix) -> dict:
    """The full metric set for one operating point.

    Read alongside :mod:`app.evaluation.semantics`:

    * ``false_accept_rate`` -- defective units that were ACCEPTED (FN / actual positive)
    * ``false_reject_rate`` -- good units that were REJECTED (FP / actual negative)

    ``false_positive_rate`` and ``false_negative_rate`` are carried as well so
    that the statistical reading of the same confusion matrix is visible in the
    same payload. Under this failure model they coincide pairwise with FRR and
    FAR respectively; the tests pin that identity so it can never drift into a
    silent contradiction.
    """
    tp, fp, tn, fn = matrix.tp, matrix.fp, matrix.tn, matrix.fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    sensitivity = recall
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, matrix.total)
    balanced_accuracy = (
        None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2.0
    )

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # frozen convention: both defined and zero is a real 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    false_accept_rate = _safe_div(fn, fn + tp)
    false_reject_rate = _safe_div(fp, fp + tn)
    acceptance_rate = _safe_div(fn + tn, matrix.total)
    rejection_rate = _safe_div(tp + fp, matrix.total)

    return {
        "sample_count": matrix.total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_reject_rate,
        "false_negative_rate": false_accept_rate,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "acceptance_rate": acceptance_rate,
        "rejection_rate": rejection_rate,
    }


def operating_point(records: Iterable[PredictionRecord]) -> dict:
    """Confusion matrix + metrics for the threshold the records were scored at.

    ``confusion_matrix`` uses the repository's frozen layout
    ``[[tn, fp], [fn, tp]]`` (ground truth on rows), matching
    ``steel_patchcore.aggregation.operating_point`` and the existing steel
    evaluation checkpoint aggregation.
    """
    rows = validate_records(records)
    matrix = confusion_matrix(rows)
    return {
        "confusion_matrix": [[matrix.tn, matrix.fp], [matrix.fn, matrix.tp]],
        "confusion": matrix.to_dict(),
        **binary_metrics(matrix),
    }


def latency_stats(records: Iterable[PredictionRecord]) -> dict:
    """Inference latency distribution over the records that carry one."""
    values = sorted(
        float(record.inference_latency_ms)
        for record in records
        if record.inference_latency_ms is not None
    )
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "max_ms": float(values[-1]),
    }


def score_distribution(records: Iterable[PredictionRecord], *, label: int | None = None) -> dict:
    """Score distribution, optionally restricted to one ground-truth class."""
    values = sorted(
        float(record.score)
        for record in records
        if label is None or record.ground_truth == label
    )
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": len(values),
        "min": float(values[0]),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "max": float(values[-1]),
    }


def score_distributions(records: Iterable[PredictionRecord]) -> dict:
    """Both class distributions plus their median gap."""
    rows = list(records)
    positive = score_distribution(rows, label=GROUND_TRUTH_POSITIVE)
    negative = score_distribution(rows, label=GROUND_TRUTH_NEGATIVE)
    gap = (
        positive["p50"] - negative["p50"]
        if positive["p50"] is not None and negative["p50"] is not None
        else None
    )
    return {
        "positive_class": positive,
        "negative_class": negative,
        "median_gap": gap,
    }


__all__ = [
    "ConfusionMatrix",
    "binary_metrics",
    "confusion_matrix",
    "latency_stats",
    "operating_point",
    "percentile",
    "score_distribution",
    "score_distributions",
]
