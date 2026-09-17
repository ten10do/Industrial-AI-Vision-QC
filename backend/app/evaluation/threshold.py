"""Threshold sweep.

Scans a set of thresholds and reports, for each one, how the operating point
moves. The output is descriptive. Nothing in this module decides that a
threshold is "production optimal", because that decision needs a business
cost function (the price of a released defect versus the price of a scrapped
good unit) and this repository has no measured cost function. The reference
points that are returned are named after the *criterion* they satisfy
(``best_f1``, ``recall_oriented``, ``conservative``) rather than after a
verdict they cannot support.

`app.mlops.quality_gate` is what turns a swept threshold into a verdict, and
it does so against a written policy, not against a heuristic in here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .records import EvaluationError, PredictionRecord, validate_records
from .metrics import operating_point

REFERENCE_POINT_NOTE = (
    "Descriptive reference points only. They are derived from this sweep, not from a "
    "measured business cost function, so none of them is a 'production optimal "
    "threshold'. Selecting a deployed threshold is a human decision recorded against "
    "a written policy."
)


def derive_threshold_grid(
    records: Iterable[PredictionRecord],
    *,
    steps: int = 21,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    """An evenly spaced grid spanning the observed score range.

    Defaults to the exact [min, max] of the recorded scores, inclusive, so the
    sweep always contains the degenerate 'reject nothing' and 'reject
    everything' ends. No rounding is applied: a rounded grid can silently skip
    the score that a real operating point sits on.
    """
    rows = validate_records(records)
    if not rows:
        raise EvaluationError("NO_RECORDS", "cannot derive a threshold grid from zero records")
    if steps < 2:
        raise EvaluationError("GRID_TOO_SMALL", f"steps must be >= 2, got {steps}")

    scores = [record.score for record in rows]
    lo = min(scores) if lower is None else float(lower)
    hi = max(scores) if upper is None else float(upper)
    if hi < lo:
        raise EvaluationError("GRID_RANGE_INVERTED", f"upper {hi} < lower {lo}")
    if hi == lo:
        return [lo]

    span = hi - lo
    grid = [lo + span * index / (steps - 1) for index in range(steps)]
    # ascending, deduplicated, order-stable
    unique: list[float] = []
    for value in grid:
        if not unique or value != unique[-1]:
            unique.append(value)
    return unique


def _sweep_row(threshold: float, rows: Sequence[PredictionRecord]) -> dict:
    scores = operating_point(record.at_threshold(threshold) for record in rows)
    return {"threshold": threshold, **scores}


def sweep_thresholds(
    records: Iterable[PredictionRecord],
    thresholds: Sequence[float] | None = None,
    *,
    steps: int = 21,
) -> list[dict]:
    """Evaluate every threshold in the grid, ascending.

    Prediction is recomputed from ``(score, threshold)`` for each row, so a
    sweep entry can never report a prediction its own score does not support.
    """
    rows = validate_records(records)
    if not rows:
        raise EvaluationError("NO_RECORDS", "cannot sweep thresholds over zero records")
    grid = list(thresholds) if thresholds is not None else derive_threshold_grid(rows, steps=steps)
    if not grid:
        raise EvaluationError("GRID_EMPTY", "threshold grid is empty")
    return [_sweep_row(float(threshold), rows) for threshold in sorted(set(float(t) for t in grid))]


def _point(row: dict) -> dict:
    return {
        "threshold": row["threshold"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "false_accept_rate": row["false_accept_rate"],
        "false_reject_rate": row["false_reject_rate"],
        "tp": row["tp"],
        "tn": row["tn"],
        "fp": row["fp"],
        "fn": row["fn"],
    }


def recommend_thresholds(
    sweep: Sequence[dict],
    *,
    false_reject_target: float = 0.0,
) -> dict:
    """Name the reference points in a sweep. Deterministic and order-stable.

    * ``best_f1``          -- highest F1; ties broken by lower false reject
      rate, then by the higher threshold.
    * ``recall_oriented``  -- the highest threshold that still attains the
      maximum recall seen anywhere in the sweep, i.e. the point that rejects
      the most while losing none of the anomalies this sweep can catch.
    * ``conservative``     -- the lowest threshold whose false reject rate is
      at or below ``false_reject_target``. ``criteria_met`` records whether
      that target was reachable at all.
    """
    if not sweep:
        raise EvaluationError("SWEEP_EMPTY", "cannot recommend thresholds from an empty sweep")

    def _key(row: dict) -> tuple:
        f1 = row["f1"]
        far = row["false_accept_rate"]
        frr = row["false_reject_rate"]
        return (f1 is not None, f1 if f1 is not None else -1.0, -(frr if frr is not None else 0.0), row["threshold"])

    best = max(sweep, key=_key)

    recalls = [row["recall"] for row in sweep if row["recall"] is not None]
    if not recalls:
        recall_point = None
        max_recall = None
    else:
        max_recall = max(recalls)
        recall_point = max(
            (row for row in sweep if row["recall"] is not None and row["recall"] == max_recall),
            key=lambda row: row["threshold"],
        )

    eligible = [
        row
        for row in sweep
        if row["false_reject_rate"] is not None
        and row["false_reject_rate"] <= false_reject_target
        and row["tp"] >= 1
    ]
    criteria_met = bool(eligible)
    if eligible:
        conservative = min(eligible, key=lambda row: row["threshold"])
    else:
        reachable = [row for row in sweep if row["false_reject_rate"] is not None]
        conservative = (
            min(reachable, key=lambda row: (row["false_reject_rate"], row["threshold"]))
            if reachable
            else None
        )

    return {
        "best_f1": _point(best),
        "recall_oriented": _point(recall_point) if recall_point is not None else None,
        "conservative": _point(conservative) if conservative is not None else None,
        "max_recall_in_sweep": max_recall,
        "false_reject_target": false_reject_target,
        "conservative_criteria_met": criteria_met,
        "note": REFERENCE_POINT_NOTE,
    }


__all__ = [
    "REFERENCE_POINT_NOTE",
    "derive_threshold_grid",
    "recommend_thresholds",
    "sweep_thresholds",
]
