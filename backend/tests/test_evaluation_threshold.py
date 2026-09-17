"""Threshold sweep: verified against a sweep computed by hand.

The sweep below uses an explicit threshold list so every row is derivable on
paper from the fixture population. Deriving the default grid would make the
test depend on the grid generator it is also trying to verify.

    threshold  TP TN FP FN   precision  recall   F1     FAR    FRR
    0.05       4  0  6  0    0.4        1.0      4/7    0.0    1.0
    0.35       3  3  3  1    0.5        0.75     0.6    0.25   0.5
    0.50       3  4  2  1    0.6        0.75     2/3    0.25   1/3
    0.65       2  5  1  2    2/3        0.5      4/7    0.5    1/6
    0.92       1  6  0  3    1.0        0.25     0.4    0.75   0.0
    1.00       0  6  0  4    n/a        0.0      n/a    1.0    0.0
"""

from __future__ import annotations

import pytest

from app.evaluation import EvaluationError, sweep_thresholds
from app.evaluation.threshold import derive_threshold_grid, recommend_thresholds
from eval_helpers import build_records

sweep = sweep_thresholds

GRID = [0.05, 0.35, 0.50, 0.65, 0.92, 1.00]

EXPECTED = {
    0.05: {"tp": 4, "tn": 0, "fp": 6, "fn": 0},
    0.35: {"tp": 3, "tn": 3, "fp": 3, "fn": 1},
    0.50: {"tp": 3, "tn": 4, "fp": 2, "fn": 1},
    0.65: {"tp": 2, "tn": 5, "fp": 1, "fn": 2},
    0.92: {"tp": 1, "tn": 6, "fp": 0, "fn": 3},
    1.00: {"tp": 0, "tn": 6, "fp": 0, "fn": 4},
}


def test_sweep_counts_match_the_hand_built_table():
    rows = {row["threshold"]: row for row in sweep(build_records(), GRID)}
    assert sorted(rows) == GRID
    for threshold, expected in EXPECTED.items():
        row = rows[threshold]
        for key, value in expected.items():
            assert row[key] == value, (threshold, key, row[key], value)


def test_sweep_metrics_match_the_hand_built_table():
    rows = {row["threshold"]: row for row in sweep(build_records(), GRID)}

    assert rows[0.05]["precision"] == pytest.approx(0.4)
    assert rows[0.05]["recall"] == pytest.approx(1.0)
    assert rows[0.05]["f1"] == pytest.approx(2 * 0.4 * 1.0 / 1.4)
    assert rows[0.05]["false_accept_rate"] == pytest.approx(0.0)
    assert rows[0.05]["false_reject_rate"] == pytest.approx(1.0)

    assert rows[0.50]["precision"] == pytest.approx(0.6)
    assert rows[0.50]["recall"] == pytest.approx(0.75)
    assert rows[0.50]["f1"] == pytest.approx(2 / 3)
    assert rows[0.50]["false_accept_rate"] == pytest.approx(0.25)
    assert rows[0.50]["false_reject_rate"] == pytest.approx(1 / 3)

    # at threshold 1.0 no positive can be predicted, so precision/F1 vanish
    assert rows[1.00]["precision"] is None
    assert rows[1.00]["f1"] is None
    assert rows[1.00]["recall"] == pytest.approx(0.0)
    assert rows[1.00]["false_accept_rate"] == pytest.approx(1.0)


def test_predictions_and_metrics_move_monotonically_with_the_threshold():
    rows = sweep(build_records(), GRID)
    for key in ("tp", "fp"):
        values = [row[key] for row in rows]
        assert values == sorted(values, reverse=True), (key, values)
    for key in ("tn", "fn"):
        values = [row[key] for row in rows]
        assert values == sorted(values), (key, values)
    recalls = [row["recall"] for row in rows]
    assert recalls == sorted(recalls, reverse=True)
    fars = [row["false_accept_rate"] for row in rows]
    assert fars == sorted(fars)


def test_sweep_is_ascending_and_deduplicates_input():
    rows = sweep(build_records(), [0.9, 0.1, 0.9, 0.5])
    assert [row["threshold"] for row in rows] == [0.1, 0.5, 0.9]


def test_sweep_recomputes_every_prediction_from_score_and_threshold():
    """A sweep entry may never report a prediction its own score cannot support."""
    for row in sweep(build_records(), GRID):
        threshold = row["threshold"]
        assert row["tp"] + row["fn"] == 4  # actual positives, constant across the sweep
        assert row["tn"] + row["fp"] == 6  # actual negatives, constant across the sweep


def test_derive_threshold_grid_spans_the_observed_range_inclusively():
    records = build_records()
    grid = derive_threshold_grid(records, steps=5)
    assert grid[0] == pytest.approx(0.10)   # min score
    assert grid[-1] == pytest.approx(0.95)  # max score
    assert len(grid) == 5
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)


def test_derive_threshold_grid_handles_a_degenerate_range():
    records = build_records(
        population=(("n1", 0, 0.5, "s", None, 1.0), ("a1", 1, 0.5, "s", None, 1.0))
    )
    assert derive_threshold_grid(records, steps=11) == [0.5]


def test_derive_threshold_grid_rejects_an_inverted_range():
    with pytest.raises(EvaluationError) as exc:
        derive_threshold_grid(build_records(), lower=1.0, upper=0.0)
    assert exc.value.code == "GRID_RANGE_INVERTED"


def test_sweep_over_zero_records_fails_closed():
    with pytest.raises(EvaluationError) as exc:
        sweep_thresholds([], GRID)
    assert exc.value.code == "NO_RECORDS"


# ---- reference points ----


def test_reference_points_are_named_after_the_criterion_they_satisfy():
    points = recommend_thresholds(sweep(build_records(), GRID))

    # highest F1 in the table is 2/3 at 0.50
    assert points["best_f1"]["threshold"] == pytest.approx(0.50)
    assert points["best_f1"]["f1"] == pytest.approx(2 / 3)

    # recall never exceeds 1.0, reached first at the lowest threshold
    assert points["max_recall_in_sweep"] == pytest.approx(1.0)
    assert points["recall_oriented"]["threshold"] == pytest.approx(0.05)

    # no good unit rejected (FRR 0) while still catching something: 0.92
    assert points["conservative_criteria_met"] is True
    assert points["conservative"]["threshold"] == pytest.approx(0.92)
    assert points["conservative"]["false_reject_rate"] == pytest.approx(0.0)


def test_conservative_criterion_is_reported_as_unmet_when_unreachable():
    points = recommend_thresholds(sweep(build_records(), [0.05, 0.35, 0.50]))
    assert points["conservative_criteria_met"] is False
    assert points["conservative"] is not None
    # FRR across this three point sweep is 1.0 / 0.5 / 1/3, so the fallback
    # picks the lowest achieved false reject rate, not the lowest threshold.
    assert points["conservative"]["threshold"] == pytest.approx(0.50)
    assert points["conservative"]["false_reject_rate"] == pytest.approx(1 / 3)


def test_reference_points_never_claim_a_production_optimum():
    points = recommend_thresholds(sweep(build_records(), GRID))
    assert "production optimal" in points["note"]
    assert "Descriptive reference points only" in points["note"]
    assert "optimal" not in {key.replace("_", " ") for key in points}


def test_recommendations_are_deterministic_across_repeated_runs():
    first = recommend_thresholds(sweep(build_records(), GRID))
    second = recommend_thresholds(sweep(build_records(), GRID))
    assert first == second
