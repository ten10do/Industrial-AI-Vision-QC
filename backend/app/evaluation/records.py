"""Prediction-level records and error typing.

Every evaluation in this repository is built from a list of
:class:`PredictionRecord`. One record is one inspected unit, and it carries
enough context to answer "what did the model see, what did it say, and what
was true" without a second lookup.

The record derives its prediction from ``(score, threshold)`` instead of
trusting a stored label. That is a deliberate constraint: it makes it
impossible for a report to claim a prediction its own score cannot support,
and it makes a threshold sweep exact rather than approximate. When a caller
supplies a prediction anyway, it is validated against the derived one and a
disagreement is a hard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

ERROR_TYPES = ("TP", "TN", "FP", "FN")

# ground truth coding shared by the whole repository
GROUND_TRUTH_NEGATIVE = 0
GROUND_TRUTH_POSITIVE = 1


class EvaluationError(ValueError):
    """A record (or a set of records) that cannot be evaluated safely."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code
        super().__init__(self.message)


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or value is None:
        raise EvaluationError(code, f"expected a finite number, got {value!r}")
    if not isinstance(value, (int, float)):
        raise EvaluationError(code, f"expected a finite number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(code, f"expected a finite number, got {value!r}")
    return number


def _label(value: Any, code: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (GROUND_TRUTH_NEGATIVE, GROUND_TRUTH_POSITIVE):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise EvaluationError(code, f"ground_truth must be 0 (negative) or 1 (positive), got {value!r}")


@dataclass(frozen=True)
class PredictionRecord:
    """One evaluated unit.

    `ground_truth` uses the repository coding: 0 = negative class
    (normal / defect-free), 1 = positive class (anomaly / defect). See
    :mod:`app.evaluation.semantics` for why that mapping is the one that
    decides accept vs reject.
    """

    sample_id: str
    ground_truth: int
    score: float
    threshold: float
    model_name: str
    model_version: str
    dataset: str | None = None
    split: str | None = None
    defect_type: str | None = None
    inference_latency_ms: float | None = None
    prediction: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise EvaluationError("SAMPLE_ID_INVALID", "sample_id must be a non-empty string")
        object.__setattr__(self, "sample_id", self.sample_id.strip())

        object.__setattr__(self, "ground_truth", _label(self.ground_truth, "GROUND_TRUTH_INVALID"))
        object.__setattr__(self, "score", _finite(self.score, "SCORE_NONFINITE"))
        object.__setattr__(self, "threshold", _finite(self.threshold, "THRESHOLD_NONFINITE"))

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise EvaluationError("MODEL_NAME_INVALID", "model_name must be a non-empty string")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise EvaluationError("MODEL_VERSION_INVALID", "model_version must be a non-empty string")

        if self.inference_latency_ms is not None:
            latency = _finite(self.inference_latency_ms, "LATENCY_NONFINITE")
            if latency < 0:
                raise EvaluationError("LATENCY_NEGATIVE", "inference_latency_ms must be >= 0")
            object.__setattr__(self, "inference_latency_ms", latency)

        derived = self.predicted_label
        if self.prediction is not None:
            declared = _label(self.prediction, "PREDICTION_INVALID")
            if declared != derived:
                raise EvaluationError(
                    "PREDICTION_SCORE_CONTRADICTION",
                    f"{self.sample_id}: prediction={declared} but score={self.score} vs "
                    f"threshold={self.threshold} derives {derived}",
                )
        else:
            object.__setattr__(self, "prediction", derived)

    # ---- derived ----

    @property
    def predicted_label(self) -> int:
        """Positive when the score reaches the threshold (>=, never >)."""
        return GROUND_TRUTH_POSITIVE if self.score >= self.threshold else GROUND_TRUTH_NEGATIVE

    @property
    def error_type(self) -> str:
        """One of TP / TN / FP / FN under the positive=positive-class coding."""
        if self.ground_truth == GROUND_TRUTH_POSITIVE:
            return "TP" if self.prediction == GROUND_TRUTH_POSITIVE else "FN"
        return "FP" if self.prediction == GROUND_TRUTH_POSITIVE else "TN"

    @property
    def is_error(self) -> bool:
        return self.error_type in ("FP", "FN")

    def at_threshold(self, threshold: float) -> "PredictionRecord":
        """The same observation re-decided at another threshold.

        Only the decision moves; the score, latency and metadata are carried
        through untouched, so a sweep can never invent evidence.
        """
        return PredictionRecord(
            sample_id=self.sample_id,
            ground_truth=self.ground_truth,
            score=self.score,
            threshold=threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            dataset=self.dataset,
            split=self.split,
            defect_type=self.defect_type,
            inference_latency_ms=self.inference_latency_ms,
        )

    # ---- serialisation ----

    def to_dict(self, *, error_type: str | None = None) -> dict:
        """Full prediction-level payload, including the error classification."""
        return {
            "sample_id": self.sample_id,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "confidence": self.score,
            "anomaly_score": self.score,
            "score": self.score,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inference_latency_ms": self.inference_latency_ms,
            "defect_type": self.defect_type,
            "dataset": self.dataset,
            "split": self.split,
            "error_type": error_type or self.error_type,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PredictionRecord":
        """Build a record from a plain mapping.

        ``score`` may arrive under any of the repository's three names
        (``score`` / ``anomaly_score`` / ``confidence``); a caller that
        supplies two of them with different values is rejected rather than
        silently averaged.
        """
        if not isinstance(row, Mapping):
            raise EvaluationError("RECORD_NOT_A_MAPPING", f"expected a mapping, got {type(row).__name__}")
        sample_id = row.get("sample_id")
        if sample_id is None:
            raise EvaluationError("SAMPLE_ID_MISSING", "record has no sample_id")

        candidates = [
            (name, row[name])
            for name in ("score", "anomaly_score", "confidence")
            if row.get(name) is not None
        ]
        if not candidates:
            raise EvaluationError("SCORE_MISSING", f"{sample_id}: record carries no score")
        first = float(candidates[0][1])
        for name, value in candidates[1:]:
            if float(value) != first:
                raise EvaluationError(
                    "SCORE_AMBIGUOUS",
                    f"{sample_id}: {candidates[0][0]}={first} disagrees with {name}={value}",
                )
        score = candidates[0][1]

        return cls(
            sample_id=sample_id,
            ground_truth=row.get("ground_truth"),
            score=score,
            threshold=row.get("threshold"),
            model_name=row.get("model_name") or "unknown",
            model_version=row.get("model_version") or "unknown",
            dataset=row.get("dataset"),
            split=row.get("split"),
            defect_type=row.get("defect_type"),
            inference_latency_ms=row.get("inference_latency_ms"),
            prediction=row.get("prediction"),
        )


def validate_records(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    """Coerce to a list and reject duplicate sample ids.

    A duplicated sample id almost always means a split leaked or a checkpoint
    was merged twice; both inflate or deflate every metric downstream, so this
    fails instead of aggregating.
    """
    out = list(records)
    for index, record in enumerate(out):
        if not isinstance(record, PredictionRecord):
            raise EvaluationError(
                "RECORD_TYPE_INVALID", f"record {index} is {type(record).__name__}, not PredictionRecord"
            )
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in out:
        if record.sample_id in seen:
            duplicates.append(record.sample_id)
        seen.add(record.sample_id)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise EvaluationError("DUPLICATE_SAMPLE_ID", f"duplicate sample ids: {preview}")
    return out


def partition_by_error_type(records: Iterable[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
    """Group records into TP / TN / FP / FN buckets, each sorted by sample id."""
    buckets: dict[str, list[PredictionRecord]] = {name: [] for name in ERROR_TYPES}
    for record in records:
        buckets[record.error_type].append(record)
    for name in ERROR_TYPES:
        buckets[name].sort(key=lambda item: item.sample_id)
    return buckets


__all__ = [
    "ERROR_TYPES",
    "GROUND_TRUTH_NEGATIVE",
    "GROUND_TRUTH_POSITIVE",
    "EvaluationError",
    "PredictionRecord",
    "partition_by_error_type",
    "validate_records",
]
