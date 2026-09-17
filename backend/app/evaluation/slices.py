"""Slice analysis over metadata that actually exists.

The discipline this module enforces: a slice is only computed for a field
that is present on the prediction records. Nothing is inferred, and nothing
is clustered into a plausible-sounding bucket. There is no lighting slice,
no camera slice, no production-line slice, no material slice and no
environment slice, because none of those fields exist on a prediction record.
Inventing them would produce a chart that looks like insight and is actually
decoration.

The dimensions that do exist:

    dataset, split, model_name, model_version, defect_type, score_range

``score_range`` is derived, and deliberately so: it splits each slice at the
operating threshold (below / at-or-above). That is the only cut with a fixed
meaning in this repository, so it is the only derived cut offered.

A slice smaller than ``min_slice_size`` is reported with its counts but its
rates stay ``None``. A rate computed from three samples is noise, and a gate
or dashboard that renders it as a percentage is lying with real numbers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .metrics import binary_metrics, confusion_matrix, latency_stats
from .records import PredictionRecord, validate_records

# Fields that exist on PredictionRecord and are therefore sliceable.
SUPPORTED_DIMENSIONS = (
    "dataset",
    "split",
    "model_name",
    "model_version",
    "defect_type",
    "score_range",
)

# Recorded here on purpose. Each entry names a slice a reader might expect and
# states why it is not offered. This is the audit trail for "we did not make
# this up".
UNSUPPORTED_DIMENSIONS = {
    "lighting": "no illumination metadata is captured on a prediction record",
    "camera": "no camera / device identity is captured on a prediction record",
    "production_line": "line identity lives on the review task, not on a model prediction",
    "station": "station identity lives on the review task, not on a model prediction",
    "material": "no material / batch composition metadata is captured per prediction",
    "environment": "no ambient condition metadata is captured per prediction",
}

SCORE_BANDS = ("below_threshold", "at_or_above_threshold")


class UnsupportedDimensionError(ValueError):
    """Requested a slice the prediction records cannot support."""


def score_band(record: PredictionRecord) -> str:
    return SCORE_BANDS[1] if record.score >= record.threshold else SCORE_BANDS[0]


def _slice_values(record: PredictionRecord, dimension: str) -> Sequence[str]:
    if dimension == "dataset":
        value = record.dataset
    elif dimension == "split":
        value = record.split
    elif dimension == "model_name":
        value = record.model_name
    elif dimension == "model_version":
        value = record.model_version
    elif dimension == "defect_type":
        value = record.defect_type
    elif dimension == "score_range":
        return [score_band(record)]
    else:
        raise UnsupportedDimensionError(
            f"{dimension!r} is not a sliceable dimension; supported: {list(SUPPORTED_DIMENSIONS)}"
        )
    # An absent metadata value is its own bucket, named explicitly. It is never
    # silently folded into a neighbouring slice.
    return [str(value) if value not in (None, "") else "(not recorded)"]


def slice_analysis(
    records: Iterable[PredictionRecord],
    *,
    dimensions: Sequence[str] = SUPPORTED_DIMENSIONS,
    min_slice_size: int = 1,
) -> dict:
    """Per-dimension breakdown of the operating point.

    Slices below ``min_slice_size`` keep their counts and confusion matrix
    (which are facts) while their derived rates are returned as ``None``
    (which is honest).
    """
    rows = validate_records(records)
    unknown = [name for name in dimensions if name not in SUPPORTED_DIMENSIONS]
    if unknown:
        raise UnsupportedDimensionError(
            f"unsupported slice dimension(s): {unknown}; supported: {list(SUPPORTED_DIMENSIONS)}"
        )
    if min_slice_size < 1:
        raise ValueError("min_slice_size must be >= 1")

    out: dict[str, dict] = {}
    for dimension in dimensions:
        buckets: dict[str, list[PredictionRecord]] = {}
        for record in rows:
            for value in _slice_values(record, dimension):
                buckets.setdefault(value, []).append(record)

        dimension_out: dict[str, dict] = {}
        for value in sorted(buckets):
            bucket = buckets[value]
            matrix = confusion_matrix(bucket)
            rates = binary_metrics(matrix) if len(bucket) >= min_slice_size else None
            entry = {
                "sample_count": len(bucket),
                "tp": matrix.tp,
                "tn": matrix.tn,
                "fp": matrix.fp,
                "fn": matrix.fn,
                "confusion_matrix": [[matrix.tn, matrix.fp], [matrix.fn, matrix.tp]],
                "latency": latency_stats(bucket),
                "rates_suppressed": rates is None,
            }
            if rates is not None:
                entry.update(rates)
            else:
                entry.update(
                    {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "false_accept_rate": None,
                        "false_reject_rate": None,
                    }
                )
            dimension_out[value] = entry

        out[dimension] = {
            "value_count": len(dimension_out),
            "min_slice_size": min_slice_size,
            "values": dimension_out,
        }

    return {
        "dimensions": out,
        "supported_dimensions": list(SUPPORTED_DIMENSIONS),
        "unsupported_dimensions": dict(UNSUPPORTED_DIMENSIONS),
    }


__all__ = [
    "SCORE_BANDS",
    "SUPPORTED_DIMENSIONS",
    "UNSUPPORTED_DIMENSIONS",
    "UnsupportedDimensionError",
    "score_band",
    "slice_analysis",
]
