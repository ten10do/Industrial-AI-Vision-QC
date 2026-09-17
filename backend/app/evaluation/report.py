"""The structured evaluation report.

One report is one offline evaluation of one model version on one dataset
split. It is the artifact that a quality gate reads, that a dashboard
renders, and that an auditor re-derives. It is therefore:

* **self-describing** -- it carries its own schema version, task semantics and
  threshold, so a reader never has to guess what "positive" meant;
* **reproducible** -- ``fingerprint`` is the SHA256 of the canonical payload
  with the clock removed, so the same inputs always produce the same digest;
* **honest about evidence** -- prediction-level failure cases are included,
  not just rates, so "the recall is 0.61" can always be followed by "and here
  are the units it missed".

Persisting the report is out of scope here; :mod:`app.services.evaluation_service`
owns storage and :mod:`app.mlops.quality_gate` owns the verdict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..mlops.attestation import canonical_json, sha256_hex
from .metrics import binary_metrics, confusion_matrix, latency_stats, score_distributions
from .records import (
    ERROR_TYPES,
    EvaluationError,
    PredictionRecord,
    partition_by_error_type,
    validate_records,
)
from .semantics import semantics_for
from .slices import SUPPORTED_DIMENSIONS, slice_analysis
from .threshold import derive_threshold_grid, recommend_thresholds, sweep_thresholds

EVALUATION_SCHEMA_VERSION = "ivqc_error_analysis_report_v1"

DEFAULT_FAILURE_CASE_LIMIT = 20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _case_rows(rows: Sequence[PredictionRecord], limit: int) -> tuple[list[dict], bool]:
    """Most diagnostic first: the confident errors, then deterministic order.

    A false alarm the model was certain about, and a missed defect it was
    almost certain about, are the two cases a reviewer learns the most from.
    Score descending wins ties by sample id, so the list never reshuffles
    between two runs over the same evidence.
    """
    ordered = sorted(rows, key=lambda record: (-record.score, record.sample_id))
    truncated = len(ordered) > limit
    return [record.to_dict() for record in ordered[:limit]], truncated


def build_evaluation_report(
    *,
    model_name: str,
    model_version: str,
    dataset: str,
    split: str,
    records: Iterable[PredictionRecord],
    threshold: float,
    task: str = "anomaly_detection",
    evaluation_time: str | None = None,
    include_threshold_analysis: bool = True,
    sweep_steps: int = 21,
    sweep_thresholds_explicit: Sequence[float] | None = None,
    slice_dimensions: Sequence[str] = SUPPORTED_DIMENSIONS,
    min_slice_size: int = 1,
    failure_case_limit: int = DEFAULT_FAILURE_CASE_LIMIT,
    extra: dict | None = None,
) -> dict:
    """Build the report. Never mutates the records it is given.

    Every record is re-decided at ``threshold`` from its own score. A caller
    that collected scores at a different threshold therefore gets a correct
    report for the threshold it names, and the disagreement is reported in
    ``threshold_diagnostics`` instead of being quietly averaged away.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise EvaluationError("MODEL_NAME_INVALID", "model_name is required")
    if not isinstance(model_version, str) or not model_version.strip():
        raise EvaluationError("MODEL_VERSION_INVALID", "model_version is required")
    if not isinstance(dataset, str) or not dataset.strip():
        raise EvaluationError("DATASET_INVALID", "dataset is required")
    if not isinstance(split, str) or not split.strip():
        raise EvaluationError("SPLIT_INVALID", "split is required")

    semantics = semantics_for(task)
    rows = validate_records(records)
    if not rows:
        raise EvaluationError("NO_RECORDS", "cannot build an evaluation report from zero records")

    threshold = float(threshold)
    if threshold != threshold or threshold in (float("inf"), float("-inf")):
        raise EvaluationError("THRESHOLD_NONFINITE", f"threshold must be finite, got {threshold!r}")

    observed_thresholds = sorted({record.threshold for record in rows})
    re_decided = observed_thresholds != [threshold]
    evaluated = [record.at_threshold(threshold) for record in rows]

    matrix = confusion_matrix(evaluated)
    metrics = binary_metrics(matrix)
    latency = latency_stats(evaluated)
    buckets = partition_by_error_type(evaluated)

    false_positive, fp_truncated = _case_rows(buckets["FP"], failure_case_limit)
    false_negative, fn_truncated = _case_rows(buckets["FN"], failure_case_limit)

    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model": {
            "name": model_name.strip(),
            "version": model_version.strip(),
        },
        "dataset": {
            "name": dataset.strip(),
            "split": split.strip(),
            "sample_count": matrix.total,
        },
        "task": semantics.task,
        "semantics": semantics.to_dict(),
        "threshold": threshold,
        "threshold_diagnostics": {
            "observed_record_thresholds": observed_thresholds,
            "records_re_decided": re_decided,
        },
        "metrics": {
            **metrics,
            "latency": latency,
        },
        "confusion_matrix": [[matrix.tn, matrix.fp], [matrix.fn, matrix.tp]],
        "confusion": matrix.to_dict(),
        "score_distributions": score_distributions(evaluated),
        "error_counts": {name: len(buckets[name]) for name in ERROR_TYPES},
        "error_rate": (matrix.fp + matrix.fn) / matrix.total if matrix.total else None,
        "failure_cases": {
            "limit": failure_case_limit,
            "counts": {
                "false_positive": matrix.fp,
                "false_negative": matrix.fn,
                "tp": matrix.tp,
                "tn": matrix.tn,
            },
            "false_positive": false_positive,
            "false_negative": false_negative,
            "truncated": {
                "false_positive": fp_truncated,
                "false_negative": fn_truncated,
            },
        },
        "slice_analysis": slice_analysis(
            evaluated, dimensions=slice_dimensions, min_slice_size=min_slice_size
        ),
        "evaluation_time": evaluation_time or _utc_now(),
    }

    if include_threshold_analysis:
        sweep = sweep_thresholds(evaluated, sweep_thresholds_explicit, steps=sweep_steps)
        report["threshold_analysis"] = {
            "grid": [row["threshold"] for row in sweep],
            "grid_size": len(sweep),
            "evaluated_threshold_in_grid": threshold in [row["threshold"] for row in sweep],
            "sweep": sweep,
            "reference_points": recommend_thresholds(sweep),
        }

    if extra:
        reserved = set(report) & set(extra)
        if reserved:
            raise EvaluationError(
                "EXTRA_SHADOWS_REPORT_FIELD", f"extra may not overwrite: {sorted(reserved)}"
            )
        report.update(extra)

    report["fingerprint"] = evaluation_fingerprint(report)
    return report


def evaluation_fingerprint(report: dict) -> str:
    """SHA256 over the canonical report with clock and digest removed.

    Removing ``evaluation_time`` is what makes two evaluations of the same
    evidence comparable: the digest answers "is this the same evaluation?"
    rather than "was this run at the same second?".
    """
    payload = {key: value for key, value in report.items() if key not in ("fingerprint", "evaluation_time")}
    return sha256_hex(canonical_json(payload))


def report_summary(report: dict) -> dict:
    """The fields a gate, a list endpoint or a dashboard cell actually needs."""
    metrics = report.get("metrics", {})
    matrix = report.get("confusion_matrix") or [[None, None], [None, None]]
    return {
        "schema_version": report.get("schema_version"),
        "fingerprint": report.get("fingerprint"),
        "model": report.get("model"),
        "dataset": report.get("dataset"),
        "task": report.get("task"),
        "threshold": report.get("threshold"),
        "sample_count": metrics.get("sample_count"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "false_accept_rate": metrics.get("false_accept_rate"),
        "false_reject_rate": metrics.get("false_reject_rate"),
        "latency_p95_ms": (metrics.get("latency") or {}).get("p95_ms"),
        "confusion_matrix": matrix,
        "fp_count": matrix[0][1],
        "fn_count": matrix[1][0],
        "evaluation_time": report.get("evaluation_time"),
    }


# Re-exported so callers can build a grid without importing the sweep module.
__all__ = [
    "DEFAULT_FAILURE_CASE_LIMIT",
    "EVALUATION_SCHEMA_VERSION",
    "build_evaluation_report",
    "derive_threshold_grid",
    "evaluation_fingerprint",
    "report_summary",
]
