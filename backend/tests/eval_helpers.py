"""Shared builders for the error-analysis and quality-gate tests.

Not a test module itself (no ``test_`` prefix), so pytest does not collect it.
The population used throughout is deliberately tiny and hand-checkable: every
expected number in the tests can be recomputed on paper from the table below,
which is the only way to know that a metric implementation is right rather
than merely self-consistent.

The fixture population (threshold 0.5 unless a test says otherwise):

    sample_id  ground_truth  score  prediction  error_type
    n1         0 (normal)    0.10   0 (accept)  TN
    n2         0 (normal)    0.20   0 (accept)  TN
    n3         0 (normal)    0.30   0 (accept)  TN
    n4         0 (normal)    0.40   0 (accept)  TN
    n5         0 (normal)    0.55   1 (reject)  FP   <- good unit rejected
    n6         0 (normal)    0.90   1 (reject)  FP   <- good unit rejected
    a1         1 (anomaly)   0.15   0 (accept)  FN   <- defective unit accepted
    a2         1 (anomaly)   0.60   1 (reject)  TP
    a3         1 (anomaly)   0.70   1 (reject)  TP
    a4         1 (anomaly)   0.95   1 (reject)  TP

    TP=3 TN=4 FP=2 FN=1
    precision = 3/5 = 0.6
    recall    = 3/4 = 0.75
    F1        = 2*0.6*0.75/1.35 = 2/3
    FAR       = FN/(FN+TP) = 1/4 = 0.25  (= 1 - recall)
    FRR       = FP/(FP+TN) = 2/6 = 1/3   (= false positive rate)
"""

from __future__ import annotations

import copy

from app.evaluation import PredictionRecord, build_evaluation_report

THRESHOLD = 0.5

# (sample_id, ground_truth, score, split, defect_type, latency_ms)
POPULATION = (
    ("n1", 0, 0.10, "test_normal", None, 10.0),
    ("n2", 0, 0.20, "test_normal", None, 11.0),
    ("n3", 0, 0.30, "test_normal", None, 12.0),
    ("n4", 0, 0.40, "test_normal", None, 13.0),
    ("n5", 0, 0.55, "test_normal", None, 14.0),
    ("n6", 0, 0.90, "test_normal", None, 15.0),
    ("a1", 1, 0.15, "test_anomaly", "class1", 20.0),
    ("a2", 1, 0.60, "test_anomaly", "class3", 21.0),
    ("a3", 1, 0.70, "test_anomaly", "class3", 22.0),
    ("a4", 1, 0.95, "test_anomaly", "class4", 23.0),
)

MODEL_NAME = "steel-patchcore-fixture"
MODEL_VERSION = "9.9.9"
DATASET = "fixture-dataset"
SPLIT = "test"

EXPECTED_TP = 3
EXPECTED_TN = 4
EXPECTED_FP = 2
EXPECTED_FN = 1


def build_records(threshold: float = THRESHOLD, *, population=POPULATION):
    return [
        PredictionRecord(
            sample_id=sample_id,
            ground_truth=ground_truth,
            score=score,
            threshold=threshold,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            dataset=DATASET,
            split=split,
            defect_type=defect_type,
            inference_latency_ms=latency,
        )
        for sample_id, ground_truth, score, split, defect_type, latency in population
    ]


def good_report(*, threshold: float = THRESHOLD, evaluation_time: str = "2026-09-17T00:00:00Z"):
    """A report that satisfies the development quality gate policy.

    Recall is 0.75 (>= 0.60), FAR is 0.25 (<= 0.40), FRR is 1/3 (> 0.10).
    Tests that need a passing FRR rule override it explicitly rather than
    editing this builder, so the default stays a fixed, meaningful point.
    """
    return build_evaluation_report(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        dataset=DATASET,
        split=SPLIT,
        records=build_records(threshold),
        threshold=threshold,
        evaluation_time=evaluation_time,
    )


def with_metrics(report: dict, **overrides) -> dict:
    """Return a copy of a report with metrics replaced and digest re-sealed.

    Re-sealing matters: the server recomputes the fingerprint, so a test that
    edits a metric and does not re-seal is testing the tamper detector, not the
    gate. Use :func:`tampered` for the detector.
    """
    from app.evaluation.report import evaluation_fingerprint

    edited = copy.deepcopy(report)
    edited["metrics"].update(overrides)
    edited.pop("fingerprint", None)
    edited["fingerprint"] = evaluation_fingerprint(edited)
    return edited


def tampered(report: dict, **overrides) -> dict:
    """A copy with metrics changed but the old fingerprint kept."""
    edited = copy.deepcopy(report)
    edited["metrics"].update(overrides)
    return edited


# ---- policy construction ----

POLICY_TEMPLATE = {
    "schema_version": "ivqc_quality_gate_policy_v1",
    "policy_id": "test-quality-gate",
    "status": "development",
    "notes": ["test policy"],
    "enforcement": {"enforced_model_types": ["patchcore"], "require_evaluation_evidence": True},
    "evidence_requirements": {
        "require_confusion_matrix": True,
        "require_failure_cases": True,
        "require_threshold_recorded": True,
        "require_sample_count": True,
    },
    "freshness": {"max_evidence_age_days": None},
    "metrics": {
        "patchcore": {
            "recall": {
                "direction": "min", "value": 0.60, "bound": 0.50,
                "source": "test fixture mirroring aggregation.DEVELOPMENT_GATE.anomaly_recall_min",
            },
            "false_accept_rate": {
                "direction": "max", "value": 0.40, "bound": 0.60,
                "source": "test fixture: dual of the 0.60 recall floor",
            },
            "false_reject_rate": {
                "direction": "max", "value": 1.0, "bound": 1.0,
                "source": "test fixture: deliberately permissive so FRR never masks another case",
            },
            "sample_count": {
                "direction": "min", "value": 10, "bound": 1,
                "source": "test fixture: the population has exactly 10 records",
            },
        }
    },
}


def policy_dict(**overrides) -> dict:
    data = copy.deepcopy(POLICY_TEMPLATE)
    for key, value in overrides.items():
        if key in ("enforcement", "evidence_requirements", "metrics") and isinstance(value, dict):
            data[key].update(value)
        else:
            data[key] = value
    return data


def write_policy(tmp_path, data: dict):
    import yaml

    path = tmp_path / "quality_gate_policy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
