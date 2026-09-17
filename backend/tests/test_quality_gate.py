"""Model Quality Gate: policy validation and the PASS / HOLD / fail-closed contract.

The gate is pure: policy in, evidence in, verdict out. Testing it here means
the API layer is never the thing that decides whether a model ships.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from app.mlops.quality_gate import (
    VERDICT_HOLD,
    VERDICT_NOT_ENFORCED,
    VERDICT_PASS,
    QualityGatePolicyError,
    evaluate_quality_gate,
    evidence_metrics,
    evidence_problems,
    load_quality_gate_policy,
)
from eval_helpers import good_report, policy_dict, tampered, with_metrics, write_policy


def _policy(tmp_path, **overrides):
    return load_quality_gate_policy(write_policy(tmp_path, policy_dict(**overrides)))


# ---- the five cases the brief names ----


def test_case_a_every_rule_satisfied_is_a_pass(tmp_path):
    report = with_metrics(good_report(), false_reject_rate=0.0)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
        timestamp="2026-09-17T00:00:00Z",
    )
    assert result.verdict == VERDICT_PASS
    assert result.passed is True
    assert result.held is False
    assert result.allows_promotion is True
    assert result.failed_rules == []
    assert result.enforced is True
    assert result.timestamp == "2026-09-17T00:00:00Z"


def test_case_b_recall_below_the_floor_is_a_hold(tmp_path):
    report = with_metrics(good_report(), recall=0.50, false_accept_rate=0.50)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.held is True
    assert "recall" in result.failed_rule_codes()
    rule = next(r for r in result.failed_rules if r["rule"] == "recall")
    assert rule["got"] == pytest.approx(0.50)
    assert rule["required"] == pytest.approx(0.60)
    assert rule["direction"] == "min"
    assert "below the required minimum" in rule["message"]
    assert rule["source"]  # a failing rule still says where its number came from


def test_case_c_false_accept_rate_above_the_ceiling_is_a_hold(tmp_path):
    report = with_metrics(good_report(), recall=0.90, false_accept_rate=0.95)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert "false_accept_rate" in result.failed_rule_codes()
    rule = next(r for r in result.failed_rules if r["rule"] == "false_accept_rate")
    assert rule["direction"] == "max"
    assert "above the allowed maximum" in rule["message"]


def test_case_d_sample_count_below_the_minimum_is_a_hold(tmp_path):
    report = with_metrics(good_report(), false_reject_rate=0.0)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path, metrics={
            "patchcore": {
                "sample_count": {
                    "direction": "min", "value": 500, "bound": 1,
                    "source": "test fixture: deliberately above the 10 sample population",
                }
            }
        }),
    )
    assert result.verdict == VERDICT_HOLD
    assert "sample_count" in result.failed_rule_codes()
    rule = next(r for r in result.failed_rules if r["rule"] == "sample_count")
    assert "below the required minimum" in rule["message"]


def test_case_e_missing_evidence_fails_closed(tmp_path):
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=None, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.failed_rule_codes() == ["evaluation_evidence_missing"]
    assert result.passed is False
    assert "cannot pass" in result.failed_rules[0]["message"]


def test_case_e_incomplete_evidence_fails_closed(tmp_path):
    report = copy.deepcopy(good_report())
    report.pop("confusion_matrix")
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert "evidence_incomplete:confusion_matrix" in result.failed_rule_codes()


def test_case_e_a_missing_metric_fails_closed_rather_than_passing_by_omission(tmp_path):
    report = copy.deepcopy(good_report())
    report["metrics"].pop("recall")
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert "metric_missing_or_invalid:recall" in result.failed_rule_codes()
    rule = next(r for r in result.failed_rules if r["rule"].startswith("metric_missing"))
    assert "cannot be checked" in rule["message"]


def test_case_e_an_unreadable_policy_fails_closed(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("schema_version: not-this-schema\n", encoding="utf-8")
    with pytest.raises(QualityGatePolicyError):
        load_quality_gate_policy(path)


# ---- HOLD must explain itself ----


def test_hold_lists_every_failed_rule_in_plain_language(tmp_path):
    report = with_metrics(good_report(), recall=0.10)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.reason.startswith("HOLD:")
    messages = [rule["message"] for rule in result.failed_rules]
    assert any("recall" in message and "below the required minimum" in message for message in messages)
    assert all(message and message[0].islower() for message in messages)


def test_the_verdict_payload_carries_what_an_auditor_needs(tmp_path):
    report = with_metrics(good_report(), false_reject_rate=0.0)
    policy = _policy(tmp_path)
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=report, policy=policy,
    )
    payload = result.to_dict()
    for key in ("verdict", "passed", "enforced", "failed_rules", "metrics", "policy", "evidence", "timestamp"):
        assert key in payload, key
    assert payload["policy"]["policy_id"] == "test-quality-gate"
    assert payload["policy"]["policy_sha256"] == policy.sha256
    assert payload["evidence"]["fingerprint"] == report["fingerprint"]
    assert payload["evidence"]["model_version"] == "9.9.9"
    assert payload["policy"]["thresholds_used"]["recall"] == 0.60


# ---- evidence integrity ----


def test_evidence_from_another_model_version_is_rejected(tmp_path):
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="1.0.0",
        evidence=good_report(), policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.failed_rule_codes() == ["evidence_version_mismatch"]


def test_evidence_from_another_model_is_rejected(tmp_path):
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="some-other-model", model_version="9.9.9",
        evidence=good_report(), policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.failed_rule_codes() == ["evidence_model_mismatch"]


def test_stale_evidence_is_rejected_when_the_policy_sets_a_freshness_limit(tmp_path):
    policy = _policy(tmp_path, freshness={"max_evidence_age_days": 30})
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=with_metrics(good_report(), false_reject_rate=0.0),
        policy=policy, evidence_age_days=31.0,
    )
    assert result.verdict == VERDICT_HOLD
    assert "evidence_stale" in result.failed_rule_codes()


def test_fresh_evidence_passes_a_freshness_limit(tmp_path):
    policy = _policy(tmp_path, freshness={"max_evidence_age_days": 30})
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=with_metrics(good_report(), false_reject_rate=0.0),
        policy=policy, evidence_age_days=1.0,
    )
    assert result.verdict == VERDICT_PASS


# ---- enforcement scope ----


def test_a_model_family_outside_the_enforced_scope_is_reported_not_hidden(tmp_path):
    result = evaluate_quality_gate(
        model_type="yolo", model_name="neu-yolov8s", model_version="1.0.0",
        evidence=None, policy=_policy(tmp_path),
    )
    assert result.verdict == VERDICT_NOT_ENFORCED
    assert result.enforced is False
    # NOT_ENFORCED is not a pass: the gate has certified nothing
    assert result.passed is False
    assert result.allows_promotion is True
    assert "does not enforce" in result.reason


def test_an_enforced_model_type_without_rules_fails_closed(tmp_path):
    result = evaluate_quality_gate(
        model_type="widgetcore", model_name="w", model_version="1.0.0",
        evidence=good_report(),
        policy=_policy(tmp_path, enforcement={
            "enforced_model_types": ["patchcore", "widgetcore"],
            "require_evaluation_evidence": True,
        }),
    )
    assert result.verdict == VERDICT_HOLD
    assert result.failed_rule_codes() == ["unknown_model_type"]


def test_an_unpinned_policy_is_surfaced_as_a_warning_not_a_silent_pass(tmp_path):
    result = evaluate_quality_gate(
        model_type="patchcore", model_name="steel-patchcore-fixture", model_version="9.9.9",
        evidence=with_metrics(good_report(), false_reject_rate=0.0),
        policy=_policy(tmp_path),
    )
    assert "policy_pin_missing" in result.warning_codes()
    assert result.policy["policy_pinned"] is False
    # pinned=false is reported on every verdict, including a passing one
    assert result.verdict == VERDICT_PASS


def test_the_shipped_policy_declares_its_enforcement_scope_and_stays_honest():
    from app.mlops.quality_gate import quality_gate_policy_path

    policy = load_quality_gate_policy(quality_gate_policy_path())
    assert policy.policy_id
    assert policy.status in ("development", "production")
    # Scope is declared, not implied. An empty list is a statement, not a bug.
    assert isinstance(policy.enforcement.enforced_model_types, tuple)
    assert policy.model_types
    for model_type in policy.model_types:
        rules = policy.rules_for(model_type)
        assert rules, model_type
        for rule in rules:
            # every shipped number must say where it came from, and that
            # includes saying "this is a placeholder"
            assert rule.source, f"{model_type}.{rule.name} has no declared source"


# ---- policy validation ----


def test_a_rule_without_a_source_is_refused(tmp_path):
    data = policy_dict()
    del data["metrics"]["patchcore"]["recall"]["source"]
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "must carry a 'source'" in str(exc.value)


def test_a_rule_naming_an_unknown_metric_is_refused(tmp_path):
    data = policy_dict()
    data["metrics"]["patchcore"]["vibes"] = {
        "direction": "min", "value": 1, "bound": 0, "source": "made up",
    }
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "not a readable evaluation metric" in str(exc.value)


def test_a_rule_that_crosses_its_own_hard_bound_is_refused(tmp_path):
    data = policy_dict()
    data["metrics"]["patchcore"]["recall"] = {
        "direction": "min", "value": 0.10, "bound": 0.50, "source": "too lax for its own bound",
    }
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "below hard floor" in str(exc.value)


def test_a_production_policy_may_not_still_cite_a_demo_number(tmp_path):
    data = policy_dict(status="production")
    data["metrics"]["patchcore"]["sample_count"]["source"] = "DEMO placeholder, replace later"
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "demo/development value" in str(exc.value)


def test_the_enforcement_scope_must_be_declared_explicitly(tmp_path):
    data = policy_dict()
    del data["enforcement"]["enforced_model_types"]
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "must be declared" in str(exc.value)


def test_an_unknown_evidence_requirement_is_refused(tmp_path):
    data = policy_dict(evidence_requirements={"require_vibes": True})
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "unknown evidence requirement" in str(exc.value)


def test_the_wrong_schema_version_is_refused(tmp_path):
    data = policy_dict(schema_version="ivqc_quality_gate_policy_v99")
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(write_policy(tmp_path, data))
    assert "schema mismatch" in str(exc.value)


def test_a_missing_policy_file_is_refused(tmp_path):
    with pytest.raises(QualityGatePolicyError) as exc:
        load_quality_gate_policy(tmp_path / "nope.yaml")
    assert "not found" in str(exc.value)


# ---- evidence helpers ----


def test_evidence_metrics_reads_only_what_is_present():
    report = good_report()
    metrics = evidence_metrics(report)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["false_accept_rate"] == pytest.approx(0.25)
    assert metrics["sample_count"] == 10
    assert metrics["latency_p95_ms"] is not None
    assert metrics["threshold"] == 0.5

    trimmed = copy.deepcopy(report)
    trimmed["metrics"].pop("latency")
    assert "latency_p95_ms" not in evidence_metrics(trimmed)


def test_evidence_problems_reports_each_structural_gap():
    requirements = {
        "require_confusion_matrix": True,
        "require_failure_cases": True,
        "require_threshold_recorded": True,
        "require_sample_count": True,
    }
    assert evidence_problems(None, requirements) == ["evidence_missing"]

    report = good_report()
    report.pop("failure_cases")
    assert evidence_problems(report, requirements) == ["evidence_incomplete:failure_cases"]

    complete = copy.deepcopy(good_report())
    assert evidence_problems(complete, requirements) == []


def test_tampering_with_a_report_breaks_its_digest():
    original = good_report()
    edited = tampered(original, precision=1.0)
    assert edited["fingerprint"] == original["fingerprint"]
    from app.evaluation.report import evaluation_fingerprint

    assert evaluation_fingerprint(edited) != edited["fingerprint"]
