"""Error Analysis Pipeline: metric arithmetic, verified by hand.

Every expected value in this file is recomputed on paper from the population
documented in ``eval_helpers``. Asserting against the implementation's own
output would only prove the code is self-consistent.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation import (
    EvaluationError,
    PredictionRecord,
    binary_metrics,
    confusion_matrix,
    latency_stats,
    operating_point,
    partition_by_error_type,
    semantics_for,
)
from app.evaluation.records import validate_records
from eval_helpers import (
    EXPECTED_FN,
    EXPECTED_FP,
    EXPECTED_TN,
    EXPECTED_TP,
    THRESHOLD,
    build_records,
)


def _approx(value, expected):
    assert value == pytest.approx(expected, abs=1e-12)


# ---- confusion matrix and error typing ----


def test_confusion_matrix_matches_the_hand_counted_population():
    matrix = confusion_matrix(build_records())
    assert (matrix.tp, matrix.tn, matrix.fp, matrix.fn) == (
        EXPECTED_TP, EXPECTED_TN, EXPECTED_FP, EXPECTED_FN
    )
    assert matrix.total == 10
    assert matrix.actual_positive == 4
    assert matrix.actual_negative == 6
    # frozen layout [[tn, fp], [fn, tp]]: ground truth on rows
    assert matrix.to_dict()["matrix"] == [[4, 2], [1, 3]]


def test_error_type_is_assigned_from_ground_truth_and_prediction():
    buckets = partition_by_error_type(build_records())
    assert [r.sample_id for r in buckets["TN"]] == ["n1", "n2", "n3", "n4"]
    assert [r.sample_id for r in buckets["FP"]] == ["n5", "n6"]
    assert [r.sample_id for r in buckets["FN"]] == ["a1"]
    assert [r.sample_id for r in buckets["TP"]] == ["a2", "a3", "a4"]
    for record in buckets["FP"]:
        assert record.ground_truth == 0 and record.prediction == 1
    for record in buckets["FN"]:
        assert record.ground_truth == 1 and record.prediction == 0


def test_prediction_is_derived_from_score_and_threshold_inclusively():
    at_boundary = PredictionRecord("edge", 1, 0.5, 0.5, "m", "1.0.0")
    assert at_boundary.prediction == 1  # score >= threshold is positive
    just_below = PredictionRecord("edge2", 1, 0.4999999, 0.5, "m", "1.0.0")
    assert just_below.prediction == 0


# ---- classification metrics ----


def test_binary_metrics_match_the_hand_computed_values():
    m = binary_metrics(confusion_matrix(build_records()))

    assert m["tp"] == 3 and m["tn"] == 4 and m["fp"] == 2 and m["fn"] == 1
    _approx(m["precision"], 3 / 5)
    _approx(m["recall"], 3 / 4)
    _approx(m["f1"], 2 / 3)
    _approx(m["sensitivity"], 3 / 4)
    _approx(m["specificity"], 4 / 6)
    _approx(m["negative_predictive_value"], 4 / 5)
    _approx(m["accuracy"], 7 / 10)
    _approx(m["balanced_accuracy"], (0.75 + 4 / 6) / 2)
    _approx(m["acceptance_rate"], 5 / 10)
    _approx(m["rejection_rate"], 5 / 10)


def test_false_accept_and_false_reject_are_the_industrial_pairing():
    """FAR is a released defect; FRR is a scrapped good unit.

    This is the identity the whole pipeline depends on, so it is asserted
    rather than assumed.
    """
    m = binary_metrics(confusion_matrix(build_records()))

    # a defective unit that was ACCEPTED: FN / actual positive
    _approx(m["false_accept_rate"], EXPECTED_FN / (EXPECTED_FN + EXPECTED_TP))
    _approx(m["false_accept_rate"], 0.25)
    _approx(m["false_accept_rate"], 1.0 - m["recall"])

    # a good unit that was REJECTED: FP / actual negative
    _approx(m["false_reject_rate"], EXPECTED_FP / (EXPECTED_FP + EXPECTED_TN))
    _approx(m["false_reject_rate"], 1 / 3)
    _approx(m["false_reject_rate"], m["false_positive_rate"])

    # and the statistical naming of the same two cells
    _approx(m["false_positive_rate"], m["false_reject_rate"])
    _approx(m["false_negative_rate"], m["false_accept_rate"])


def test_far_and_recall_are_complements_at_every_operating_point():
    for threshold in (0.05, 0.35, 0.5, 0.65, 0.92, 1.0):
        m = binary_metrics(confusion_matrix(build_records(threshold)))
        _approx(m["false_accept_rate"], 1.0 - m["recall"])
        _approx(
            m["false_reject_rate"],
            m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else 0.0,
        )


def test_undefined_rates_are_none_not_zero():
    """No positive samples at all: recall and F1 are undefined."""
    records = build_records(population=(("n1", 0, 0.1, "s", None, 1.0), ("n2", 0, 0.9, "s", None, 1.0)))
    m = binary_metrics(confusion_matrix(records))
    assert m["recall"] is None
    assert m["f1"] is None
    assert m["false_accept_rate"] is None
    # FP and TN are still fully defined
    assert m["false_reject_rate"] == pytest.approx(0.5)


def test_zero_precision_and_recall_produce_a_real_zero_f1():
    records = build_records(
        population=(("n1", 0, 0.9, "s", None, 1.0), ("a1", 1, 0.1, "s", None, 1.0))
    )
    m = binary_metrics(confusion_matrix(records))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0  # frozen convention, not None


# ---- operating point payload ----


def test_operating_point_exposes_matrix_latency_and_sample_count():
    op = operating_point(build_records())
    assert op["confusion_matrix"] == [[4, 2], [1, 3]]
    assert op["sample_count"] == 10
    _approx(op["precision"], 0.6)
    _approx(op["recall"], 0.75)
    _approx(op["false_accept_rate"], 0.25)
    _approx(op["false_reject_rate"], 1 / 3)
    # every field the brief requires at an operating point
    for key in (
        "tp", "tn", "fp", "fn", "precision", "recall", "f1",
        "false_accept_rate", "false_reject_rate", "confusion_matrix", "sample_count",
    ):
        assert key in op, key


def test_latency_stats_are_computed_over_records_that_carry_latency():
    stats = latency_stats(build_records())
    # sorted latencies: 10,11,12,13,14,15,20,21,22,23
    expected_mean = (10 + 11 + 12 + 13 + 14 + 15 + 20 + 21 + 22 + 23) / 10
    assert stats["count"] == 10
    _approx(stats["mean_ms"], expected_mean)
    # p50 interpolates between the 5th and 6th values (0.5 * 9 = 4.5)
    assert stats["p50_ms"] == pytest.approx(14.5)
    # p95 interpolates between the 9th and 10th values (0.95 * 9 = 8.55)
    assert stats["p95_ms"] == pytest.approx(22.55)
    assert stats["max_ms"] == 23.0


def test_latency_stats_are_null_when_no_record_reports_latency():
    records = [
        PredictionRecord("n1", 0, 0.1, 0.5, "m", "1.0.0"),
        PredictionRecord("a1", 1, 0.9, 0.5, "m", "1.0.0"),
    ]
    stats = latency_stats(records)
    assert stats["count"] == 0
    assert all(stats[key] is None for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"))


# ---- record validation ----


def test_records_reject_duplicate_sample_ids():
    rows = build_records()
    with pytest.raises(EvaluationError) as exc:
        validate_records(rows + [rows[0]])
    assert exc.value.code == "DUPLICATE_SAMPLE_ID"


def test_records_reject_a_prediction_the_score_cannot_support():
    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("x", 1, 0.1, 0.5, "m", "1.0.0", prediction=1)
    assert exc.value.code == "PREDICTION_SCORE_CONTRADICTION"


def test_records_reject_non_finite_scores_and_bad_labels():
    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("x", 1, math.nan, 0.5, "m", "1.0.0")
    assert exc.value.code == "SCORE_NONFINITE"

    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("x", 2, 0.5, 0.5, "m", "1.0.0")
    assert exc.value.code == "GROUND_TRUTH_INVALID"

    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("x", 1, 0.5, float("inf"), "m", "1.0.0")
    assert exc.value.code == "THRESHOLD_NONFINITE"


def test_records_reject_a_missing_identity():
    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("   ", 1, 0.5, 0.5, "m", "1.0.0")
    assert exc.value.code == "SAMPLE_ID_INVALID"

    with pytest.raises(EvaluationError) as exc:
        PredictionRecord("x", 1, 0.5, 0.5, "m", "  ")
    assert exc.value.code == "MODEL_VERSION_INVALID"


def test_records_accept_the_repository_score_aliases():
    record = PredictionRecord.from_mapping({
        "sample_id": "x", "ground_truth": 1, "anomaly_score": 0.7, "threshold": 0.5,
        "model_name": "m", "model_version": "1",
    })
    assert record.score == 0.7
    assert record.prediction == 1

    with pytest.raises(EvaluationError) as exc:
        PredictionRecord.from_mapping({
            "sample_id": "x", "ground_truth": 1, "score": 0.7, "confidence": 0.9,
            "threshold": 0.5, "model_name": "m", "model_version": "1",
        })
    assert exc.value.code == "SCORE_AMBIGUOUS"


# ---- semantics ----


def test_task_semantics_make_positive_the_thing_that_cannot_ship():
    anomaly = semantics_for("patchcore")
    assert anomaly.task == "anomaly_detection"
    assert anomaly.positive_label == "anomaly"
    assert anomaly.positive_decision == "REJECT"
    assert anomaly.negative_decision == "ACCEPT"

    defect = semantics_for("yolo")
    assert defect.task == "defect_detection"
    assert defect.positive_label == "defect"


def test_unknown_task_fails_loudly_instead_of_guessing():
    with pytest.raises(Exception) as exc:
        semantics_for("some-new-model-family")
    assert "no declared label semantics" in str(exc.value)
