"""Phase 8: MLOps API integration (registry, monitoring, feedback, drift).

The registry contract changed: identity and privileged facts are separate
calls, both authenticated. Registration no longer accepts metrics or a domain
verdict; those arrive through the signed attestation endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mlops.attestation import (  # noqa: E402
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    attestation_payload,
    sign_attestation,
)

MODEL = "neu-yolov8s-e2e"
SECRET = "test-pipeline-secret"

IDENTITY = {
    "model_name": MODEL,
    "model_version": "1.0.0",
    "model_type": "yolo",
    "artifact_uri": "inference-service/models/best.pt",
    "dataset_version": "neu-det-yolo-v1",
    "training_run_id": "run-1",
}

METRICS = {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5}


async def _attest(client, auth, entry, artifact, eval_report, *, metrics=None, domain_validated=True):
    metrics = dict(metrics or METRICS)
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
    return await client.post(
        f"/api/v1/models/{entry['id']}/attest",
        json=body,
        headers={**auth("pipeline"), SIGNATURE_HEADER: signature, TIMESTAMP_HEADER: str(ts)},
    )


async def _promote(client, auth, entry_id, **extra):
    body = {"required_domain": "steel", "approved_by": "qa-manager", "reason": "mlops api test"}
    body.update(extra)
    return await client.post(f"/api/v1/models/{entry_id}/promote", json=body, headers=auth("approver"))


async def test_register_attest_gate_promote(client, db_session, auth, artifact, eval_report):
    r = await client.post("/api/v1/models", json=IDENTITY, headers=auth("engineer"))
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["status"] == "CANDIDATE"
    assert entry["metrics"] == {}

    a = await _attest(client, auth, entry, artifact, eval_report)
    assert a.status_code == 200, a.text
    assert a.json()["provenance"]["artifact_hash_verified"] is True
    assert a.json()["provenance"]["domain_evidence_verified"] is True

    g = await client.post(
        f"/api/v1/models/{entry['id']}/gate",
        json={"required_domain": "steel"},
        headers=auth("viewer"),
    )
    assert g.status_code == 200
    assert g.json()["gate"]["passed"] is True

    p = await _promote(client, auth, entry["id"])
    assert p.status_code == 200, p.text
    assert p.json()["status"] == "PRODUCTION"

    prod = await client.get(f"/api/v1/models/production/{MODEL}", headers=auth("viewer"))
    assert prod.status_code == 200
    assert prod.json()["model_version"] == "1.0.0"


async def test_duplicate_version_rejected(client, db_session, auth):
    await client.post("/api/v1/models", json=IDENTITY, headers=auth("engineer"))
    r = await client.post("/api/v1/models", json=IDENTITY, headers=auth("engineer"))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "duplicate_version"


async def test_promote_rejected_when_domain_not_validated(client, db_session, auth, artifact, eval_report):
    body = {**IDENTITY, "model_name": "mvtec-patchcore-e2e", "model_type": "patchcore"}
    r = await client.post("/api/v1/models", json=body, headers=auth("engineer"))
    entry = r.json()
    # perfect AUROC but the domain evidence is absent -> gate rejects (8F)
    a = await _attest(
        client, auth, entry, artifact, eval_report,
        metrics={"image_auroc": 1.0, "pixel_auroc": 0.986, "latency_ms": 755.0},
        domain_validated=False,
    )
    assert a.status_code == 200, a.text
    assert a.json()["provenance"]["domain_evidence_verified"] is False

    g = await client.post(
        f"/api/v1/models/{entry['id']}/gate",
        json={"required_domain": "steel"},
        headers=auth("viewer"),
    )
    assert g.json()["gate"]["passed"] is False
    assert any("domain" in b for b in g.json()["gate"]["blocked"])

    p = await _promote(client, auth, entry["id"])
    assert p.status_code == 422
    assert p.json()["error"]["code"] == "promotion_gate_failed"


async def test_rollback_switches_production(client, db_session, auth, artifact, eval_report):
    v1 = (await client.post("/api/v1/models", json=IDENTITY, headers=auth("engineer"))).json()
    await _attest(client, auth, v1, artifact, eval_report)
    await _promote(client, auth, v1["id"])

    v2 = (await client.post(
        "/api/v1/models", json={**IDENTITY, "model_version": "2.0.0"}, headers=auth("engineer")
    )).json()
    await _attest(client, auth, v2, artifact, eval_report)
    await _promote(client, auth, v2["id"])

    rb = await client.post(
        "/api/v1/models/rollback",
        json={"model_name": MODEL, "model_version": "1.0.0",
              "approved_by": "qa-manager", "reason": "rollback acceptance test"},
        headers=auth("approver"),
    )
    assert rb.status_code == 200, rb.text
    prod = await client.get(f"/api/v1/models/production/{MODEL}", headers=auth("viewer"))
    assert prod.json()["model_version"] == "1.0.0"


async def test_model_metrics_and_human_feedback_endpoints(client, db_session):
    m = await client.get("/api/v1/model-metrics")
    assert m.status_code == 200
    assert "inference_count" in m.json()

    f = await client.get("/api/v1/human-feedback")
    assert f.status_code == 200
    assert "defect_confirmation_rate" in f.json()

    d = await client.get("/api/v1/drift")
    assert d.status_code == 200
    assert d.json()["overall"] in ("NORMAL", "WARNING", "CRITICAL")
    assert "data drift only" in d.json()["note"]


async def test_invalid_monitoring_dates_return_422(client):
    for path in ("/api/v1/model-metrics", "/api/v1/human-feedback"):
        response = await client.get(path, params={"time_from": "not-a-date"})
        assert response.status_code == 422


async def test_training_candidate_source_identity_fields(client, db_session, auth):
    """Semantic fix (Phase 9): dataset / model / deployment identities are
    distinct fields on the retraining candidate manifest."""
    cands = (await client.get("/api/v1/training-candidates", headers=auth("release-manager"), params={"kind": "all"})).json()
    for c in cands:
        assert "source_dataset_version" in c
        assert "source_model_version" in c
        assert "source_deployment_version" in c
        assert "dataset_version" not in c
        assert "model_version" not in c
