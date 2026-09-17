"""Lifecycle integration: a quality gate HOLD must block promotion.

These tests drive the real HTTP layer, because the guarantee being tested is
that *no code path* reaches PRODUCTION while the quality gate holds. A
service-level test could pass while the router quietly skipped the gate.
"""

from __future__ import annotations

import copy

import pytest

from app.config import get_settings
from app.evaluation.report import evaluation_fingerprint
from app.mlops import quality_gate as quality_gate_module
from app.mlops.attestation import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    attestation_payload,
    evaluation_attestation_payload,
    sign_attestation,
)
from app.services.registry_service import RegistryService
from eval_helpers import (
    DATASET,
    MODEL_NAME,
    MODEL_VERSION,
    SPLIT,
    build_records,
    good_report,
    policy_dict,
    with_metrics,
    write_policy,
)

SECRET = "test-pipeline-secret"
MODEL = MODEL_NAME

# Metrics that satisfy the shipped promotion policy for patchcore.
PROMOTABLE_METRICS = {"image_auroc": 0.95, "pixel_auroc": 0.95, "latency_ms": 800.0}

IDENTITY = {
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
    "model_type": "patchcore",
    "artifact_uri": "inference-service/models/best.pt",
    "dataset_version": "fixture-dataset-v1",
    "training_run_id": "fixture-run",
}


@pytest.fixture
def enforced_policy(tmp_path, monkeypatch):
    """Point the process at a policy that actually enforces patchcore."""
    path = write_policy(
        tmp_path,
        policy_dict(
            enforcement={"enforced_model_types": ["patchcore"], "require_evaluation_evidence": True},
        ),
    )
    monkeypatch.setenv("IVQC_QUALITY_GATE_POLICY_PATH", str(path))
    monkeypatch.setenv("IVQC_QUALITY_GATE_POLICY_SHA256", "")
    get_settings.cache_clear()
    quality_gate_module.reset_quality_gate_policy_cache()
    try:
        yield path
    finally:
        get_settings.cache_clear()
        quality_gate_module.reset_quality_gate_policy_cache()


@pytest.fixture
def unenforced_policy(tmp_path, monkeypatch):
    """Point the process at a policy that declares no enforced model family."""
    path = write_policy(tmp_path, policy_dict(enforcement={"enforced_model_types": []}))
    monkeypatch.setenv("IVQC_QUALITY_GATE_POLICY_PATH", str(path))
    monkeypatch.setenv("IVQC_QUALITY_GATE_POLICY_SHA256", "")
    get_settings.cache_clear()
    quality_gate_module.reset_quality_gate_policy_cache()
    try:
        yield path
    finally:
        get_settings.cache_clear()
        quality_gate_module.reset_quality_gate_policy_cache()


# ---- helpers ----


async def _register(client, auth, **overrides):
    body = {**IDENTITY, **overrides}
    r = await client.post("/api/v1/models", json=body, headers=auth("engineer"))
    assert r.status_code == 200, r.text
    return r.json()


async def _attest(client, auth, entry, artifact, eval_report, metrics=None, domain_validated=True):
    evidence = None
    if domain_validated:
        evidence = {
            "domain": "steel",
            "dataset_version": "fixture-dataset-v1",
            "eval_report_uri": eval_report["uri"],
            "eval_report_sha256": eval_report["sha256"],
            "validated_by": "eval-pipeline",
        }
    body = {
        "artifact_sha256": artifact["sha256"],
        "metrics": PROMOTABLE_METRICS if metrics is None else metrics,
        "domain_validated": domain_validated,
        "domain_evidence": evidence,
    }
    payload = attestation_payload(
        model_name=entry["model_name"],
        model_version=entry["model_version"],
        training_run_id=entry.get("training_run_id"),
        body=body,
    )
    signature, ts = sign_attestation(SECRET, payload)
    return await client.post(
        f"/api/v1/models/{entry['id']}/attest",
        json=body,
        headers={**auth("pipeline"), SIGNATURE_HEADER: signature, TIMESTAMP_HEADER: str(ts)},
    )


async def _submit_evaluation(client, auth, entry, report, *, sign=True, tamper_with=None):
    """Mirror the endpoint's signing contract: the pipeline signs what it sends.

    The payload is built from the *report's* own model identity, because the
    report is the pipeline's statement about which version it evaluated. The
    server then enforces that this identity matches the registry entry.
    """
    body = {"report": copy.deepcopy(report)}
    if tamper_with is not None:
        tamper_with(body["report"])
    model = body["report"].get("model") or {}
    payload = evaluation_attestation_payload(
        model_name=str(model.get("name") or entry["model_name"]),
        model_version=str(model.get("version") or entry["model_version"]),
        report_sha256=str(body["report"].get("fingerprint") or ""),
    )
    headers = dict(auth("pipeline"))
    if sign:
        signature, ts = sign_attestation(SECRET, payload)
        headers[SIGNATURE_HEADER] = signature
        headers[TIMESTAMP_HEADER] = str(ts)
    return await client.post(
        f"/api/v1/models/{entry['id']}/evaluations", json=body, headers=headers
    )


async def _promote(client, auth, entry_id, *, role="approver", approved_by="qa-manager",
                   reason="lifecycle gate test"):
    body = {"required_domain": "steel", "approved_by": approved_by, "reason": reason}
    return await client.post(
        f"/api/v1/models/{entry_id}/promote", json=body, headers=auth(role)
    )


async def _status(client, auth, entry_id):
    r = await client.get(f"/api/v1/models/{entry_id}", headers=auth("viewer"))
    assert r.status_code == 200, r.text
    return r.json()["status"]


def _hold_report():
    """Recall 0.10: the gate must HOLD."""
    return with_metrics(good_report(), recall=0.10, false_accept_rate=0.90, false_reject_rate=0.0)


def _pass_report():
    return with_metrics(good_report(), false_reject_rate=0.0)


# ---- HOLD blocks promotion ----


async def test_quality_gate_hold_blocks_promotion_and_is_audited(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    assert (await _attest(client, auth, entry, artifact, eval_report)).status_code == 200
    submitted = await _submit_evaluation(client, auth, entry, _hold_report())
    assert submitted.status_code == 200, submitted.text

    response = await _promote(client, auth, entry["id"])
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "quality_gate_held"
    assert "recall" in response.json()["error"]["message"]

    # the candidate did not move
    assert await _status(client, auth, entry["id"]) == "CANDIDATE"

    svc = RegistryService()
    trail = await svc.audit_trail(db_session, __import__("uuid").UUID(entry["id"]))
    promote_rows = [row for row in trail if row.action == "promote"]
    assert len(promote_rows) == 1
    row = promote_rows[0]
    assert row.outcome == "DENIED"
    assert row.payload["block"] == "quality_gate_hold"
    assert row.payload["quality_gate"]["verdict"] == "HOLD"
    assert "recall" in row.payload["quality_gate"]["failed_rule_codes"]
    assert row.payload["quality_gate"]["policy"]["policy_id"] == "test-quality-gate"
    assert row.actor == "tester-approver"


async def test_human_approval_cannot_bypass_a_held_gate(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    await _submit_evaluation(client, auth, entry, _hold_report())

    # a perfectly valid, distinct approver with a proper reason
    response = await _promote(client, auth, entry["id"], approved_by="plant-qa-director",
                              reason="approved by the quality director after review")
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "quality_gate_held"
    assert await _status(client, auth, entry["id"]) == "CANDIDATE"


async def test_missing_evidence_holds_when_the_policy_enforces_the_model_family(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)

    response = await _promote(client, auth, entry["id"])
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "quality_gate_held"
    assert "evaluation_evidence_missing" in response.json()["error"]["message"]
    assert await _status(client, auth, entry["id"]) == "CANDIDATE"


# ---- PASS still needs the existing human approval ----


async def test_a_passing_gate_still_requires_human_approval(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    assert (await _submit_evaluation(client, auth, entry, _pass_report())).status_code == 200

    gate = await client.get(f"/api/v1/models/{entry['id']}/quality-gate", headers=auth("viewer"))
    assert gate.json()["quality_gate"]["verdict"] == "PASS"
    assert gate.json()["quality_gate"]["passed"] is True

    # no approver identity at all
    r = await client.post(
        f"/api/v1/models/{entry['id']}/promote",
        json={"required_domain": "steel", "approved_by": None, "reason": "no approver"},
        headers=auth("approver"),
    )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "approval_required"

    # self approval
    r = await _promote(client, auth, entry["id"], approved_by="tester-approver")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "self_approval_forbidden"

    # engineer cannot promote at all
    r = await _promote(client, auth, entry["id"], role="engineer")
    assert r.status_code == 403

    # a real approver, a real reason
    r = await _promote(client, auth, entry["id"])
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "PRODUCTION"
    assert r.json()["quality_gate"]["verdict"] == "PASS"
    assert await _status(client, auth, entry["id"]) == "PRODUCTION"


async def test_an_unenforced_scope_records_the_exemption_in_the_journal(
    client, db_session, auth, artifact, eval_report, unenforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)

    response = await _promote(client, auth, entry["id"])
    assert response.status_code == 200, response.text
    assert response.json()["quality_gate"]["verdict"] == "NOT_ENFORCED"
    assert response.json()["quality_gate"]["passed"] is False

    svc = RegistryService()
    trail = await svc.audit_trail(db_session, __import__("uuid").UUID(entry["id"]))
    promote_row = next(row for row in trail if row.action == "promote" and row.outcome == "APPLIED")
    assert promote_row.payload["quality_gate_verdict"] == "NOT_ENFORCED"
    assert promote_row.payload["quality_gate"]["policy"]["enforcement"]["enabled"] is False


# ---- evidence submission is a privileged, signed path ----


async def test_submitting_evidence_requires_the_pipeline_role(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    r = await _submit_evaluation(client, auth, entry, _pass_report())
    assert r.status_code == 200

    body = {"report": _pass_report()}
    for role in ("engineer", "approver", "viewer", "operator"):
        r = await client.post(
            f"/api/v1/models/{entry['id']}/evaluations", json=body, headers=auth(role)
        )
        assert r.status_code == 403, (role, r.text)


async def test_submitting_evidence_requires_authentication(
    client_unauthenticated, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client_unauthenticated, auth)
    await _attest(client_unauthenticated, auth, entry, artifact, eval_report)
    r = await client_unauthenticated.post(
        f"/api/v1/models/{entry['id']}/evaluations", json={"report": _pass_report()}
    )
    assert r.status_code == 401


async def test_an_unsigned_report_is_rejected(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    r = await _submit_evaluation(client, auth, entry, _pass_report(), sign=False)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "evaluation_attestation_invalid"
    assert "signature_missing" in r.json()["error"]["message"]


async def test_a_report_edited_after_sealing_is_rejected(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    """Second line of defence: the server recomputes the digest itself."""
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)

    def tamper(report):
        report["metrics"]["recall"] = 0.99   # fingerprint left untouched

    r = await _submit_evaluation(client, auth, entry, _pass_report(), tamper_with=tamper)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "report_fingerprint_mismatch"


async def test_a_confusion_matrix_that_disagrees_with_its_own_counts_is_rejected(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)

    def tamper(report):
        report["metrics"]["tp"] = 99
        report.pop("fingerprint", None)
        report["fingerprint"] = evaluation_fingerprint(report)

    r = await _submit_evaluation(client, auth, entry, _pass_report(), tamper_with=tamper)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "report_confusion_matrix_inconsistent"


async def test_evidence_for_a_different_model_version_is_rejected(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)

    other = good_report()
    other["model"] = {"name": MODEL_NAME, "version": "0.0.1"}
    other.pop("fingerprint")
    other["fingerprint"] = evaluation_fingerprint(other)

    r = await _submit_evaluation(client, auth, entry, other)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "report_model_mismatch"


# ---- reads ----


async def test_stored_evidence_is_listable_and_readable(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    report = _pass_report()
    assert (await _submit_evaluation(client, auth, entry, report)).status_code == 200

    listing = await client.get(f"/api/v1/models/{entry['id']}/evaluations", headers=auth("viewer"))
    assert listing.status_code == 200, listing.text
    payload = listing.json()
    assert payload["count"] == 1
    item = payload["evaluations"][0]
    assert item["model_version"] == MODEL_VERSION
    assert item["dataset"] == {"name": DATASET, "split": SPLIT}
    assert item["sample_count"] == 10
    assert item["confusion_matrix"] == [[4, 2], [1, 3]]
    assert item["report_sha256"] == report["fingerprint"]
    assert item["metrics"]["recall"] == pytest.approx(0.75)

    detail = await client.get(f"/api/v1/evaluations/{item['id']}", headers=auth("viewer"))
    assert detail.status_code == 200
    assert detail.json()["summary"]["fp_count"] == 2
    assert detail.json()["summary"]["fn_count"] == 1
    assert detail.json()["summary"]["confusion_matrix"] == [[4, 2], [1, 3]]


async def test_the_dry_run_reports_both_gates_and_writes_nothing(
    client, db_session, auth, artifact, eval_report, enforced_policy
):
    entry = await _register(client, auth)
    await _attest(client, auth, entry, artifact, eval_report)
    await _submit_evaluation(client, auth, entry, _hold_report())

    before = len(await RegistryService().audit_trail(db_session, __import__("uuid").UUID(entry["id"])))

    r = await client.post(
        f"/api/v1/models/{entry['id']}/gate", json={"required_domain": "steel"}, headers=auth("viewer")
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gate"]["passed"] is True            # the promotion gate is fine
    assert body["quality_gate"]["verdict"] == "HOLD"  # the quality gate is not
    assert body["promotion_allowed"] is False

    after = len(await RegistryService().audit_trail(db_session, __import__("uuid").UUID(entry["id"])))
    assert after == before, "a dry run must not write to the governance journal"


async def test_the_policy_endpoint_says_whether_it_enforces_anything(
    client, auth, enforced_policy
):
    r = await client.get("/api/v1/model-quality-gate/policy", headers=auth("viewer"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["enforcement_enabled"] is True
    assert body["policy"]["policy_id"] == "test-quality-gate"
    assert body["policy"]["policy_sha256"]
    assert "HOLD" in body["verdicts"]


async def test_the_shipped_policy_reports_that_it_enforces_nothing_yet(client, auth):
    """The repository default must be visible, not silent."""
    get_settings.cache_clear()
    quality_gate_module.reset_quality_gate_policy_cache()
    try:
        r = await client.get("/api/v1/model-quality-gate/policy", headers=auth("viewer"))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is True
        assert body["enforcement_enabled"] is False
        assert body["policy"]["enforcement"]["enforced_model_types"] == []
    finally:
        get_settings.cache_clear()
        quality_gate_module.reset_quality_gate_policy_cache()
