"""Structured evaluation report: shape, failure cases, slices, reproducibility."""

from __future__ import annotations

import copy

import pytest

from app.evaluation import EvaluationError, build_evaluation_report, evaluation_fingerprint
from app.evaluation.slices import (
    SUPPORTED_DIMENSIONS,
    UNSUPPORTED_DIMENSIONS,
    UnsupportedDimensionError,
    slice_analysis,
)
from eval_helpers import (
    DATASET,
    MODEL_NAME,
    MODEL_VERSION,
    SPLIT,
    build_records,
    good_report,
)

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "model",
    "dataset",
    "task",
    "semantics",
    "threshold",
    "metrics",
    "confusion_matrix",
    "failure_cases",
    "slice_analysis",
    "evaluation_time",
    "fingerprint",
)


def test_report_carries_every_field_the_brief_requires():
    report = good_report()
    for key in REQUIRED_TOP_LEVEL:
        assert key in report, key

    assert report["model"] == {"name": MODEL_NAME, "version": MODEL_VERSION}
    assert report["dataset"] == {"name": DATASET, "split": SPLIT, "sample_count": 10}
    assert report["threshold"] == 0.5

    metrics = report["metrics"]
    for key in (
        "precision", "recall", "f1", "false_accept_rate", "false_reject_rate", "latency",
        "tp", "tn", "fp", "fn", "sample_count",
    ):
        assert key in metrics, key

    assert report["confusion_matrix"] == [[4, 2], [1, 3]]
    assert metrics["sample_count"] == 10
    assert metrics["latency"]["p95_ms"] is not None


def test_report_states_its_own_task_semantics():
    report = good_report()
    assert report["task"] == "anomaly_detection"
    assert report["semantics"]["positive_label"] == "anomaly"
    assert report["semantics"]["positive_decision"] == "REJECT"


def test_failure_cases_name_the_actual_samples_not_just_counts():
    report = good_report()
    cases = report["failure_cases"]

    assert cases["counts"] == {"false_positive": 2, "false_negative": 1, "tp": 3, "tn": 4}
    assert [row["sample_id"] for row in cases["false_positive"]] == ["n6", "n5"]
    assert [row["sample_id"] for row in cases["false_negative"]] == ["a1"]

    for row in cases["false_positive"]:
        assert row["error_type"] == "FP"
        assert row["ground_truth"] == 0
        assert row["prediction"] == 1
        assert row["threshold"] == 0.5
        assert row["anomaly_score"] == row["confidence"]
    assert cases["false_negative"][0]["error_type"] == "FN"
    assert cases["false_negative"][0]["defect_type"] == "class1"


def test_failure_cases_are_ordered_by_confidence_then_id():
    """The most confident error is the most diagnostic, so it comes first."""
    cases = good_report()["failure_cases"]["false_positive"]
    assert [row["sample_id"] for row in cases] == ["n6", "n5"]
    assert cases[0]["anomaly_score"] > cases[1]["anomaly_score"]


def test_failure_cases_are_truncated_deterministically():
    report = build_evaluation_report(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
        records=build_records(), threshold=0.5, failure_case_limit=1,
    )
    assert [row["sample_id"] for row in report["failure_cases"]["false_positive"]] == ["n6"]
    assert report["failure_cases"]["truncated"]["false_positive"] is True
    assert report["failure_cases"]["truncated"]["false_negative"] is False
    # the count is still the true count, not the truncated length
    assert report["failure_cases"]["counts"]["false_positive"] == 2


def test_report_is_reproducible_regardless_of_when_it_was_built():
    first = good_report(evaluation_time="2026-01-01T00:00:00Z")
    second = good_report(evaluation_time="2026-12-31T23:59:59Z")
    assert first["fingerprint"] == second["fingerprint"]
    assert first["evaluation_time"] != second["evaluation_time"]


def test_the_fingerprint_moves_when_the_evidence_moves():
    base = good_report()
    assert evaluation_fingerprint(base) == base["fingerprint"]

    edited = copy.deepcopy(base)
    edited["metrics"]["recall"] = 0.5
    edited.pop("fingerprint")
    assert evaluation_fingerprint(edited) != base["fingerprint"]


def test_report_re_decides_records_scored_at_another_threshold():
    records = build_records(threshold=0.05)  # collected at a different threshold
    report = build_evaluation_report(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
        records=records, threshold=0.5,
    )
    assert report["threshold_diagnostics"]["records_re_decided"] is True
    assert report["threshold_diagnostics"]["observed_record_thresholds"] == [0.05]
    # at 0.5 the counts are the 0.5 operating point, not the 0.05 one
    assert report["confusion_matrix"] == [[4, 2], [1, 3]]


def test_report_marks_records_that_already_agreed_with_the_threshold():
    report = build_evaluation_report(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
        records=build_records(threshold=0.5), threshold=0.5,
    )
    assert report["threshold_diagnostics"]["records_re_decided"] is False


def test_report_refuses_an_empty_population():
    with pytest.raises(EvaluationError) as exc:
        build_evaluation_report(
            model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
            records=[], threshold=0.5,
        )
    assert exc.value.code == "NO_RECORDS"


def test_report_refuses_an_unknown_task_rather_than_assuming_one():
    with pytest.raises(Exception):
        build_evaluation_report(
            model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
            records=build_records(), threshold=0.5, task="quantum-anomaly",
        )


def test_report_refuses_extra_fields_that_would_shadow_its_own():
    with pytest.raises(EvaluationError) as exc:
        build_evaluation_report(
            model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
            records=build_records(), threshold=0.5, extra={"metrics": {}},
        )
    assert exc.value.code == "EXTRA_SHADOWS_REPORT_FIELD"


def test_report_can_be_built_without_the_threshold_sweep():
    report = build_evaluation_report(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, dataset=DATASET, split=SPLIT,
        records=build_records(), threshold=0.5, include_threshold_analysis=False,
    )
    assert "threshold_analysis" not in report
    assert report["metrics"]["recall"] == pytest.approx(0.75)


# ---- slices ----


def test_slice_analysis_breaks_down_the_operating_point_per_dimension():
    slices = slice_analysis(build_records())
    assert set(slices["dimensions"]) == set(SUPPORTED_DIMENSIONS)

    split_values = slices["dimensions"]["split"]["values"]
    assert set(split_values) == {"test_normal", "test_anomaly"}
    assert split_values["test_normal"]["sample_count"] == 6
    assert split_values["test_normal"]["fp"] == 2
    assert split_values["test_normal"]["tp"] == 0
    assert split_values["test_anomaly"]["fn"] == 1
    assert split_values["test_anomaly"]["tp"] == 3


def test_defect_type_slice_uses_only_recorded_metadata():
    slices = slice_analysis(build_records())
    values = slices["dimensions"]["defect_type"]["values"]
    assert set(values) == {"class1", "class3", "class4", "(not recorded)"}
    assert values["class3"]["sample_count"] == 2
    assert values["class3"]["tp"] == 2
    assert values["class1"]["fn"] == 1
    # normals carry no defect type, and that absence is its own labelled bucket
    assert values["(not recorded)"]["sample_count"] == 6


def test_score_range_slice_splits_at_the_operating_threshold():
    slices = slice_analysis(build_records())
    values = slices["dimensions"]["score_range"]["values"]
    assert set(values) == {"below_threshold", "at_or_above_threshold"}
    # below 0.5: n1(0.10) n2(0.20) n3(0.30) n4(0.40) a1(0.15)
    assert values["below_threshold"]["sample_count"] == 5
    # at or above 0.5: n5(0.55) n6(0.90) a2(0.60) a3(0.70) a4(0.95)
    assert values["at_or_above_threshold"]["sample_count"] == 5
    assert values["below_threshold"]["fn"] == 1     # a1 was missed
    assert values["at_or_above_threshold"]["fp"] == 2  # n5, n6 were false alarms


def test_small_slices_report_counts_but_suppress_rates():
    slices = slice_analysis(build_records(), dimensions=("defect_type",), min_slice_size=3)
    values = slices["dimensions"]["defect_type"]["values"]

    small = values["class1"]  # one sample
    assert small["sample_count"] == 1
    assert small["fn"] == 1                       # the count is a fact
    assert small["recall"] is None                # the rate is withheld
    assert small["rates_suppressed"] is True

    medium = values["class3"]  # two samples, still under the floor
    assert medium["sample_count"] == 2
    assert medium["rates_suppressed"] is True

    big = values["(not recorded)"]  # six normals carry no defect type
    assert big["sample_count"] == 6
    assert big["rates_suppressed"] is False
    assert big["false_reject_rate"] == pytest.approx(2 / 6)


def test_slice_analysis_refuses_a_dimension_the_records_cannot_support():
    for forbidden in UNSUPPORTED_DIMENSIONS:
        with pytest.raises(UnsupportedDimensionError):
            slice_analysis(build_records(), dimensions=(forbidden,))


def test_slice_analysis_declares_which_dimensions_it_refuses_and_why():
    slices = slice_analysis(build_records())
    assert set(slices["unsupported_dimensions"]) == set(UNSUPPORTED_DIMENSIONS)
    for dimension, reason in slices["unsupported_dimensions"].items():
        assert reason.strip(), dimension
    # the specific ones the brief names as off-limits
    for forbidden in ("lighting", "camera", "production_line", "material", "environment"):
        assert forbidden in slices["unsupported_dimensions"]
