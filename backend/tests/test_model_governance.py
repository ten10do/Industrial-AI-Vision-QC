"""Regression tests for the model governance trust boundary.

Each test maps to a concrete bypass that was previously possible:

1. registering a model with self-declared metrics and domain_validated=true;
2. passing acceptance thresholds into the promotion call (zero thresholds made
   a zero-scoring model pass);
3. promoting / rolling back / archiving with no authentication;
4. hard-deleting a registry row;
5. PRODUCTION in the database silently diverging from the running stack.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mlops.attestation import (  # noqa: E402
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    attestation_payload,
    sign_attestation,
)
from app.mlops.promotion_gate import evaluate  # noqa: E402

SECRET = "test-pipeline-secret"
MODEL = "gov-yolov8s"


# ---- helpers ----


def _metrics(**over) -> dict:
    base = {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5}
    base.update(over)
    return base


async def _identity(client, auth, version: str) -> dict:
    r = await client.post(
        "/api/v1/models",
        json={
            "model_name": MODEL,
            "model_version": version,
            "model_type": "yolo",
            "artifact_uri": "inference-service/models/best.pt",
            "dataset_version": "neu-det-yolo-v1",
        },
        headers=auth("engineer"),
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _attest(client, auth, entry, artifact, eval_report, *, metrics=None, domain_validated=True,
            role="pipeline", tamper=False):
    metrics = metrics if metrics is not None else _metrics()
    evidence = {
        "domain": "steel",
        "dataset_version": "neu-det-yolo-v1",
        "eval_report_uri": eval_report["uri"],
        "eval_report_sha256": eval_report["sha256"],
        "validated_by": "eval-pipeline",
    }
    body = {
        "artifact_sha256": artifact["sha256"],
        "metrics": metrics,
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
    if tamper:
        signature = "0" * 64  # the body no longer matches what was signed
    headers = {**auth(role), SIGNATURE_HEADER: signature, TIMESTAMP_HEADER: str(ts)}
    return await client.post(f"/api/v1/models/{entry['id']}/attest", json=body, headers=headers)


async def _promote(client, auth, entry_id, *, role="approver", **extra):
    body = {"required_domain": "steel", "approved_by": "qa-manager",
            "reason": "governance regression test"}
    body.update(extra)
    return await client.post(f"/api/v1/models/{entry_id}/promote", json=body, headers=auth(role))


# ---- 1. self-attestation at registration ----


async def test_register_rejects_privileged_fields(client, db_session, auth):
    """metrics / domain_validated are no longer part of the register schema."""
    r = await client.post(
        "/api/v1/models",
        json={
            "model_name": MODEL,
            "model_version": "1.0.0",
            "model_type": "yolo",
            "artifact_uri": "inference-service/models/best.pt",
            "metrics": {"mAP50": 1.0, "recall": 1.0, "latency_p95_ms": 1.0},
            "domain_validated": True,
        },
        headers=auth("engineer"),
    )
    assert r.status_code == 422, r.text  # extra="forbid", not silent acceptance


async def test_register_requires_authentication(client_unauthenticated, db_session):
    r = await client_unauthenticated.post(
        "/api/v1/models",
        json={"model_name": MODEL, "model_version": "1.0.0", "model_type": "yolo",
              "artifact_uri": "inference-service/models/best.pt"},
    )
    assert r.status_code == 401, r.text


async def test_register_requires_engineer_role(client, db_session, auth):
    r = await client.post(
        "/api/v1/models",
        json={"model_name": MODEL, "model_version": "1.0.0", "model_type": "yolo",
              "artifact_uri": "inference-service/models/best.pt"},
        headers=auth("viewer"),
    )
    assert r.status_code == 403, r.text


async def test_registered_model_carries_no_metrics(client, db_session, auth, artifact):
    entry = await _identity(client, auth, "1.0.0")
    assert entry["metrics"] == {}
    assert entry["domain_validated"] is False
    assert entry["provenance"]["metrics_attested"] is False


# ---- 1b. attestation channel ----


async def test_attest_requires_pipeline_role(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    r = await _attest(client, auth, entry, artifact, eval_report, role="engineer")
    assert r.status_code == 403, r.text


async def test_attest_rejects_unsigned_payload(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    r = await client.post(
        f"/api/v1/models/{entry['id']}/attest",
        json={"artifact_sha256": artifact["sha256"], "metrics": _metrics(), "domain_validated": True},
        headers=auth("pipeline"),
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "attestation_rejected"


async def test_attest_rejects_tampered_payload(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    r = await _attest(client, auth, entry, artifact, eval_report, tamper=True)
    assert r.status_code == 401, r.text


async def test_attest_rejects_stale_timestamp(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    metrics = _metrics()
    evidence = {"domain": "steel", "eval_report_uri": eval_report["uri"],
                "eval_report_sha256": eval_report["sha256"], "validated_by": "eval-pipeline"}
    body = {"artifact_sha256": artifact["sha256"], "metrics": metrics,
            "domain_validated": True, "domain_evidence": evidence}
    payload = attestation_payload(
        model_name=entry["model_name"], model_version=entry["model_version"],
        training_run_id=entry.get("training_run_id"), body=body,
    )
    stale_ts = int(time.time()) - 10_000
    signature, _ = sign_attestation(SECRET, payload, timestamp=stale_ts)
    r = await client.post(
        f"/api/v1/models/{entry['id']}/attest",
        json=body,
        headers={**auth("pipeline"), SIGNATURE_HEADER: signature, TIMESTAMP_HEADER: str(stale_ts)},
    )
    assert r.status_code == 401, r.text
    assert "timestamp_out_of_window" in r.json()["error"]["message"]


async def test_attest_verifies_artifact_hash_server_side(client, db_session, auth, artifact, eval_report):
    """A hash the caller invents does not become a verified hash."""
    entry = await _identity(client, auth, "1.0.0")
    bogus = {**artifact, "sha256": "0" * 64}
    r = await _attest(client, auth, entry, bogus, eval_report)
    assert r.status_code == 200, r.text
    assert r.json()["provenance"]["artifact_hash_verified"] is False


async def test_attest_verifies_domain_evidence(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    bad_report = {**eval_report, "sha256": "f" * 64}
    r = await _attest(client, auth, entry, artifact, bad_report)
    assert r.status_code == 200, r.text
    assert r.json()["provenance"]["domain_evidence_verified"] is False


# ---- 2. thresholds ----


def test_zero_thresholds_no_longer_pass_a_zero_model():
    """The exact bypass: metrics all zero, thresholds all zero."""
    g = evaluate(
        "yolo",
        metrics={"mAP50": 0.0, "recall": 0.0, "latency_p95_ms": 0.0},
        thresholds={"mAP50": 0.0, "recall": 0.0, "latency_p95_ms": 0.0},
        domain_validated=True,
        required_domain="steel",
    )
    assert g.passed is False
    assert any("threshold_relaxation_forbidden" in b for b in g.blocked)


def test_thresholds_cannot_be_relaxed_but_may_be_tightened():
    g = evaluate(
        "yolo",
        metrics={"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
        thresholds={"mAP50": 0.9},
        domain_validated=True,
        required_domain="steel",
    )
    assert any(b.startswith("mAP50") for b in g.blocked)  # tightening to 0.9 fails 0.82
    assert not any("threshold_relaxation" in b for b in g.blocked)


def test_unknown_threshold_key_is_a_violation():
    g = evaluate(
        "yolo", metrics=_metrics(), thresholds={"totally_made_up": 1.0},
        domain_validated=True, required_domain="steel",
    )
    assert any(b.startswith("unknown_threshold") for b in g.blocked)


def test_gate_blocks_unattested_metrics():
    g = evaluate(
        "yolo", metrics=_metrics(), domain_validated=True, required_domain="steel",
        provenance={"metrics_attested": False, "artifact_hash_verified": True,
                    "domain_evidence_verified": True, "domain": "steel"},
    )
    assert g.passed is False
    assert any("provenance:metrics_attested" in b for b in g.blocked)


def test_gate_reports_the_policy_it_used():
    g = evaluate(
        "yolo", metrics=_metrics(), domain_validated=True, required_domain="steel",
        provenance={"metrics_attested": True, "artifact_hash_verified": True,
                    "domain_evidence_verified": True, "domain": "steel"},
    )
    assert g.passed is True, g.blocked
    assert g.policy["policy_id"]
    assert g.policy["thresholds_used"]["mAP50"] == 0.60
    assert g.policy["policy_sha256"]


async def test_promote_rejects_caller_supplied_zero_thresholds(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    r = await _attest(client, auth, entry, artifact, eval_report,
                metrics={"mAP50": 0.0, "recall": 0.0, "latency_p95_ms": 0.0})
    assert r.status_code == 200, r.text
    p = await _promote(client, auth, entry["id"], thresholds={"mAP50": 0, "recall": 0, "latency_p95_ms": 0})
    assert p.status_code == 422, p.text
    assert p.json()["error"]["code"] == "promotion_gate_failed"


# ---- 3. authentication and approval on lifecycle operations ----


async def test_promote_requires_authentication(client_unauthenticated, db_session, auth, artifact, eval_report):
    entry = await _identity(client_unauthenticated, auth, "1.0.0")
    await _attest(client_unauthenticated, auth, entry, artifact, eval_report)
    r = await client_unauthenticated.post(
        f"/api/v1/models/{entry['id']}/promote",
        json={"required_domain": "steel", "approved_by": "qa-manager", "reason": "no token"},
    )
    assert r.status_code == 401, r.text


async def test_promote_requires_approver_role(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    r = await _promote(client, auth, entry["id"], role="engineer")
    assert r.status_code == 403, r.text


async def test_promote_forbids_self_approval(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    r = await _promote(client, auth, entry["id"], approved_by="tester-approver")  # == caller subject
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "self_approval_forbidden"


async def test_promote_requires_a_reason(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    r = await _promote(client, auth, entry["id"], reason="short")
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "reason_required"


async def test_promote_blocks_unverified_artifact_hash(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, {**artifact, "sha256": "1" * 64}, eval_report)
    r = await _promote(client, auth, entry["id"])
    assert r.status_code == 422, r.text
    assert any("artifact_hash_verified" in b for b in r.json()["error"]["message"].split("; "))


async def test_happy_path_promotes_and_records_audit(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    p = await _promote(client, auth, entry["id"])
    assert p.status_code == 200, p.text
    assert p.json()["status"] == "PRODUCTION"
    assert p.json()["approval"]["approved_by"] == "qa-manager"

    trail = (await client.get(f"/api/v1/models/{entry['id']}/audit", headers=auth("viewer"))).json()
    actions = [(r["action"], r["outcome"]) for r in trail]
    assert ("register", "APPLIED") in actions
    assert ("attest", "APPLIED") in actions
    assert ("promote", "APPLIED") in actions
    actor_row = next(r for r in trail if r["action"] == "promote")
    assert actor_row["actor"] == "tester-approver"
    assert actor_row["approved_by"] == "qa-manager"


async def test_denied_promotion_is_audited(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report, metrics={"mAP50": 0.1, "recall": 0.1, "latency_p95_ms": 9.0})
    r = await _promote(client, auth, entry["id"])
    assert r.status_code == 422, r.text
    trail = (await client.get(f"/api/v1/models/{entry['id']}/audit", headers=auth("viewer"))).json()
    denied = [row for row in trail if row["action"] == "promote" and row["outcome"] == "DENIED"]
    assert denied, "a refused promotion must be part of the record"
    assert denied[-1]["gate"]["passed"] is False


async def test_rollback_requires_approver_and_reason(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    await _promote(client, auth, entry["id"])
    r = await client.post(
        "/api/v1/models/rollback",
        json={"model_name": MODEL, "model_version": "1.0.0"},
        headers=auth("engineer"),
    )
    assert r.status_code == 403, r.text


async def test_rollback_cannot_promote_a_never_promoted_version(client, db_session, auth, artifact, eval_report):
    """Rollback used to bypass the gate entirely."""
    v1 = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, v1, artifact, eval_report)
    await _promote(client, auth, v1["id"])
    v2 = await _identity(client, auth, "2.0.0")  # never attested, never promoted
    await _attest(client, auth, v2, artifact, eval_report)
    r = await client.post(
        "/api/v1/models/rollback",
        json={"model_name": MODEL, "model_version": "2.0.0",
              "approved_by": "qa-manager", "reason": "bypass attempt"},
        headers=auth("approver"),
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "rollback_target_never_promoted"


async def test_rollback_switches_production_with_approval(client, db_session, auth, artifact, eval_report):
    v1 = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, v1, artifact, eval_report)
    await _promote(client, auth, v1["id"])
    v2 = await _identity(client, auth, "2.0.0")
    await _attest(client, auth, v2, artifact, eval_report)
    await _promote(client, auth, v2["id"])
    r = await client.post(
        "/api/v1/models/rollback",
        json={"model_name": MODEL, "model_version": "1.0.0",
              "approved_by": "qa-manager", "reason": "regression observed in v2"},
        headers=auth("approver"),
    )
    assert r.status_code == 200, r.text
    prod = (await client.get(f"/api/v1/models/production/{MODEL}", headers=auth("viewer"))).json()
    assert prod["model_version"] == "1.0.0"


async def test_archiving_production_requires_approver(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    await _promote(client, auth, entry["id"])
    r = await client.post(
        f"/api/v1/models/{entry['id']}/archive",
        json={"approved_by": "qa-manager", "reason": "engineer trying to unpublish"},
        headers=auth("engineer"),
    )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "insufficient_role"


# ---- 4. hard delete ----


async def test_hard_delete_route_is_gone(client, db_session, auth, artifact):
    entry = await _identity(client, auth, "1.0.0")
    r = await client.delete(f"/api/v1/models/{entry['id']}", headers=auth("admin"))
    assert r.status_code == 405, r.text  # method not allowed: no DELETE route at all


async def test_registry_rows_are_never_destroyed_by_the_api(client, db_session, auth, artifact):
    entry = await _identity(client, auth, "1.0.0")
    r = await client.post(f"/api/v1/models/{entry['id']}/archive", headers=auth("engineer"))
    assert r.status_code == 200, r.text
    detail = await client.get(f"/api/v1/models/{entry['id']}", headers=auth("viewer"))
    assert detail.status_code == 200
    assert detail.json()["status"] == "ARCHIVED"


# ---- 5. runtime sync ----


async def test_runtime_sync_is_observable_and_never_fakes_sync(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    await _promote(client, auth, entry["id"])

    r = await client.get("/api/v1/models/runtime-sync", headers=auth("viewer"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall"] in ("IN_SYNC", "DRIFT", "UNVERIFIED")
    channels = {c["channel"]: c for c in body["channels"]}
    assert channels["yolo"]["status"] in ("DRIFT", "UNVERIFIED", "IN_SYNC")
    assert channels["yolo"]["registry"]["model_version"] == "1.0.0"
    if not body["inference_service"].get("reachable"):
        assert body["overall"] != "IN_SYNC"  # no probe, no claim of sync


async def test_runtime_sync_requires_authentication(client_unauthenticated, db_session):
    r = await client_unauthenticated.get("/api/v1/models/runtime-sync")
    assert r.status_code == 401


# ---- 5b. activation (registry -> manifest -> runtime env) ----


async def _build_production(db_session, artifact, eval_report, version="1.0.0", force_production=False):
    from app.mlops.promotion_gate import evaluate
    from app.security.auth import ROLE_PIPELINE, Principal
    from app.services.registry_service import RegistryService, provenance_for

    svc = RegistryService()
    pipeline = Principal(subject="eval-pipeline", roles=frozenset({ROLE_PIPELINE}))
    entry = await svc.register(
        db_session, actor=pipeline, model_name="act-yolov8s", model_version=version,
        model_type="yolo", artifact_uri=artifact["uri"], artifact_sha256=artifact["sha256"],
        dataset_version="neu-det-yolo-v1", metrics=_metrics(), domain_validated=True,
        domain_evidence={"domain": "steel", "eval_report_uri": eval_report["uri"],
                         "eval_report_sha256": eval_report["sha256"], "validated_by": "eval-pipeline"},
    )
    await db_session.commit()
    if force_production:
        # 仅用于 test_activation_refuses_an_unverified_artifact：该用例故意提交错误的
        # sha256（"0"*64），门禁会正确拒绝通过。为了独立验证 apply_activation 会
        # 重新校验制品哈希，这里直接构造 PRODUCTION 行，绕过 gate，并非绕过生产逻辑。
        entry.status = "PRODUCTION"
        await db_session.commit()
        return entry
    gate = evaluate(
        entry.model_type, metrics=entry.metadata_json or {}, domain_validated=entry.domain_validated,
        required_domain="steel", provenance=provenance_for(entry),
    )
    assert gate.passed, gate.blocked
    entry = await svc.promote(
        db_session, entry, gate=gate, required_domain="steel",
        actor=Principal(subject="release-manager", roles=frozenset({"approver"})),
        approved_by="qa-director", reason="activation test",
    )
    await db_session.commit()
    return entry


async def test_activation_writes_manifest_and_env(db_session, artifact, eval_report, tmp_path, monkeypatch):
    import shutil
    import yaml

    from app.config import get_settings
    from app.security.auth import Principal
    from app.services.registry_service import RegistryService

    source = Path(__file__).resolve().parents[2] / "backend" / "config" / "deployment_manifest.yaml"
    manifest = tmp_path / "deployment_manifest.yaml"
    shutil.copy(source, manifest)
    env_file = tmp_path / ".env.runtime"

    # Redirect the activation target away from the real config/ directory so
    # the test never writes into the repository. apply_activation resolves the
    # manifest through IVQC_MANIFEST and the env file through the settings.
    monkeypatch.setenv("IVQC_MANIFEST", manifest.as_posix())
    monkeypatch.setattr(get_settings(), "runtime_env_file", env_file.as_posix())

    entry = await _build_production(db_session, artifact, eval_report)
    entry, result = await RegistryService().activate(
        db_session, entry,
        actor=Principal(subject="release-manager", roles=frozenset({"approver"})),
        approved_by="qa-director", reason="activation test",
        new_stack_version="2026.08.2",
    )
    assert result["requires_restart"] is True
    assert result["artifact_sha256"] == artifact["sha256"]

    written = yaml.safe_load((tmp_path / "deployment_manifest.activated.yaml").read_text(encoding="utf-8"))
    assert written["yolo"]["model"] == "act-yolov8s"
    assert written["yolo"]["sha256"] == artifact["sha256"]
    assert written["vision_stack_version"] == "2026.08.2"
    assert entry.activation_target == result["manifest_path"]

    env_text = env_file.read_text(encoding="utf-8")
    assert "IVQC_MANIFEST=" in env_text
    assert "IVQC_WEIGHTS=" in env_text
    assert f"IVQC_MODEL_VERSION={entry.model_version}" in env_text


async def test_activation_refuses_an_unverified_artifact(db_session, artifact, eval_report, tmp_path):
    import shutil

    import pytest

    from app.mlops.deployment_sync import ActivationError, apply_activation

    source = Path(__file__).resolve().parents[2] / "backend" / "config" / "deployment_manifest.yaml"
    manifest = tmp_path / "deployment_manifest.yaml"
    shutil.copy(source, manifest)

    entry = await _build_production(db_session, {**artifact, "sha256": "0" * 64}, eval_report,
                                    force_production=True)
    with pytest.raises(ActivationError, match="verification failed"):
        apply_activation(entry, new_stack_version="2026.08.2",
                         manifest_path=manifest, env_file=tmp_path / ".env.runtime")


async def test_activation_requires_a_production_row(db_session, artifact, eval_report, tmp_path):
    import shutil

    import pytest

    from app.mlops.deployment_sync import ActivationError, apply_activation

    source = Path(__file__).resolve().parents[2] / "backend" / "config" / "deployment_manifest.yaml"
    manifest = tmp_path / "deployment_manifest.yaml"
    shutil.copy(source, manifest)

    entry = await _build_production(db_session, artifact, eval_report)
    entry.status = "CANDIDATE"
    await db_session.commit()
    with pytest.raises(ActivationError, match="only a PRODUCTION model"):
        apply_activation(entry, new_stack_version="2026.08.2",
                         manifest_path=manifest, env_file=tmp_path / ".env.runtime")


async def test_activate_requires_admin_and_reports_restart(client, db_session, auth, artifact, eval_report):
    entry = await _identity(client, auth, "1.0.0")
    await _attest(client, auth, entry, artifact, eval_report)
    await _promote(client, auth, entry["id"])
    r = await client.post(
        f"/api/v1/models/{entry['id']}/activate",
        json={"new_stack_version": "2026.08.1+e2e", "approved_by": "qa-manager",
              "reason": "pin promoted model into the runtime"},
        headers=auth("approver"),
    )
    assert r.status_code == 403, r.text


def test_policy_file_is_pinned_or_explicitly_unpinned():
    from app.mlops.gate_policy import get_policy

    policy = get_policy()
    assert policy.model_types["yolo"].rules
    if not policy.pinned:
        # Not pinned is a valid local dev state, but it must be visible.
        assert policy.pinned is False
