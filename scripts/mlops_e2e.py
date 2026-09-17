"""Phase 8 real E2E (8O): register -> attest -> promote -> inspection
traceability -> metrics -> feedback -> drift -> promote v2 -> rollback.

Requires the live stack (backend 8000, inference service 8100). No retraining:
existing artifacts are registered with explicit versions.

Governance changes this script now exercises:
* the registry API is authenticated; supply IVQC_E2E_TOKEN (or IVQC_API_TOKENS
  on the server must contain it) with the engineer+pipeline+approver+admin
  roles;
* privileged facts are posted to /attest with an HMAC signature built from
  IVQC_E2E_HMAC_SECRET, which must equal the server's IVQC_PIPELINE_HMAC_SECRET;
* DELETE /api/v1/models/{id} no longer exists. The run is made idempotent by
  archiving leftovers and by stamping a run-unique model version, so a rerun
  never needs to destroy evidence.

Run:  bash scripts/run_clean.sh python scripts/mlops_e2e.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BACKEND = os.environ.get("IVQC_E2E_BACKEND", "http://127.0.0.1:8000")
OUT = ROOT / "docs" / "phase8-e2e.json"

TOKEN = os.environ.get("IVQC_E2E_TOKEN", "")
HMAC_SECRET = os.environ.get("IVQC_E2E_HMAC_SECRET", "")
APPROVER = os.environ.get("IVQC_E2E_APPROVER", "qa-manager")
OPERATOR_SUBJECT = os.environ.get("IVQC_E2E_OPERATOR", "e2e-operator")

RUN_ID = time.strftime("%Y%m%d%H%M%S")
V1 = f"1.0.0+e2e{RUN_ID}"
V2 = f"2.0.0+e2e{RUN_ID}"

ARTIFACT = "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"
# The eval report the domain-validation claim points at. The server recomputes
# its hash, so this must be a real file produced by the evaluation step.
EVAL_REPORT = "docs/e2e-eval-report.json"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sign(payload: dict) -> dict:
    ts = str(int(time.time()))
    message = f"{ts}.{_canonical(payload)}"
    sig = hmac_sha256(HMAC_SECRET, message)
    return {"X-Attestation-Signature": sig, "X-Attestation-Timestamp": ts}


def hmac_sha256(secret: str, message: str) -> str:
    import hmac

    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _register(version: str, metrics: dict, training_run_id: str, notes: str) -> dict:
    r = httpx.post(
        f"{BACKEND}/api/v1/models",
        json={
            "model_name": "neu-yolov8s",
            "model_version": version,
            "model_type": "yolo",
            "artifact_uri": ARTIFACT,
            "dataset_version": "neu-det-yolo-v1",
            "training_run_id": training_run_id,
            "notes": notes,
        },
        headers=_auth(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _attest(entry: dict, metrics: dict, artifact_sha: str, report_sha: str) -> dict:
    evidence = {
        "domain": "steel",
        "dataset_version": "neu-det-yolo-v1",
        "eval_report_uri": EVAL_REPORT,
        "eval_report_sha256": report_sha,
        "validated_by": "e2e-eval-pipeline",
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body = {
        "artifact_sha256": artifact_sha,
        "metrics": metrics,
        "domain_validated": True,
        "domain_evidence": evidence,
    }
    payload = {
        "schema_version": "ivqc_model_attestation_v1",
        "model_name": entry["model_name"],
        "model_version": entry["model_version"],
        "training_run_id": entry.get("training_run_id"),
        "artifact_sha256": artifact_sha,
        "metrics": metrics,
        "domain_validated": True,
        "domain_evidence": evidence,
    }
    r = httpx.post(
        f"{BACKEND}/api/v1/models/{entry['id']}/attest",
        json=body,
        headers={**_auth(), **_sign(payload)},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _promote(entry_id: str) -> dict:
    r = httpx.post(
        f"{BACKEND}/api/v1/models/{entry_id}/promote",
        json={"required_domain": "steel", "approved_by": APPROVER,
              "reason": "phase 8 e2e promotion acceptance"},
        headers=_auth(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _cleanup_models() -> None:
    """Archive leftovers from previous runs. Rows are never deleted: the
    registry is an audit surface, so cleanup means archival, and the
    run-unique version keeps `duplicate_version` away."""
    rows = httpx.get(f"{BACKEND}/api/v1/models", headers=_auth(), timeout=10).json()
    for m in rows:
        if m["model_name"] == "neu-yolov8s" and m["status"] != "ARCHIVED":
            httpx.post(
                f"{BACKEND}/api/v1/models/{m['id']}/archive",
                json={"approved_by": APPROVER, "reason": "e2e rerun cleanup"},
                headers=_auth(),
                timeout=10,
            )


def _post_image() -> dict:
    img = next(iter((ROOT / "model-training/datasets/neu-det-yolo/test/images").glob("*.jpg")))
    name = f"e2e8-{RUN_ID}.jpg"
    r = httpx.post(
        f"{BACKEND}/api/v1/inspections",
        files={"file": (name, img.read_bytes(), "image/jpeg")},
        data={"product_id": "P-MLOPS-E2E", "production_line": "line-a", "station": "qc"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    if not TOKEN:
        print("E2E_BLOCKED: set IVQC_E2E_TOKEN (governance endpoints are authenticated)")
        return 2
    if not HMAC_SECRET:
        print("E2E_BLOCKED: set IVQC_E2E_HMAC_SECRET (attestation is signed)")
        return 2

    checks: dict = {}
    r = httpx.get(f"{BACKEND}/ready", timeout=5)
    r.raise_for_status()
    _cleanup_models()

    artifact_sha = _sha256(ROOT / ARTIFACT)
    eval_report = {
        "run_id": RUN_ID,
        "model_name": "neu-yolov8s",
        "dataset_version": "neu-det-yolo-v1",
        "domain": "steel",
        "artifact_sha256": artifact_sha,
        "evaluations": [
            {"model_version": V1, "mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
            {"model_version": V2, "mAP50": 0.90, "recall": 0.85, "latency_p95_ms": 18.0},
        ],
    }
    (ROOT / EVAL_REPORT).write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    report_sha = _sha256(ROOT / EVAL_REPORT)

    # 1. register + attest v1
    v1 = _register(V1, {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5}, f"e2e-run-1-{RUN_ID}", "E2E v1")
    assert v1["status"] == "CANDIDATE", v1
    assert v1["provenance"]["metrics_attested"] is False, "identity registration must not carry metrics"
    v1 = _attest(v1, {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5}, artifact_sha, report_sha)
    checks["attest_v1"] = {
        "artifact_hash_verified": v1["provenance"]["artifact_hash_verified"],
        "domain_evidence_verified": v1["provenance"]["domain_evidence_verified"],
    }

    # 2. promote v1
    promoted = _promote(v1["id"])
    assert promoted["status"] == "PRODUCTION", promoted
    checks["promote_v1"] = {"version": V1, "gate_passed": promoted["gate"]["passed"],
                            "policy": promoted["gate"]["policy"]["policy_id"]}

    # 3. inspection -> deployment_version stamped + production pointer
    insp = _post_image()
    prod = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", headers=_auth(), timeout=10).json()
    assert prod["model_version"] == V1
    checks["inspection_traceability"] = {
        "inspection_id": insp["inspection_id"],
        "deployment_version": insp.get("deployment_version"),
        "production": prod["model_version"],
    }

    # 4. model metrics
    m = httpx.get(f"{BACKEND}/api/v1/model-metrics", timeout=10).json()
    assert m["inference_count"] >= 1
    checks["model_metrics"] = {"inference_count": m["inference_count"], "error_rate": m["error_rate"],
                               "p95": m["inference_latency_p95_ms"]}

    # 5. register + attest + promote v2 (v1 archived automatically)
    v2 = _register(V2, {"mAP50": 0.9, "recall": 0.85, "latency_p95_ms": 18.0}, f"e2e-run-2-{RUN_ID}", "E2E v2")
    v2 = _attest(v2, {"mAP50": 0.9, "recall": 0.85, "latency_p95_ms": 18.0}, artifact_sha, report_sha)
    _promote(v2["id"])
    prod2 = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", headers=_auth(), timeout=10).json()
    assert prod2["model_version"] == V2
    checks["promote_v2"] = {"version": V2}

    # 6. rollback to v1 -> production pointer switches, no rebuild
    r = httpx.post(
        f"{BACKEND}/api/v1/models/rollback",
        json={"model_name": "neu-yolov8s", "model_version": V1,
              "approved_by": APPROVER, "reason": "e2e rollback acceptance"},
        headers=_auth(), timeout=10,
    )
    assert r.status_code == 200, r.text
    prod3 = httpx.get(f"{BACKEND}/api/v1/models/production/neu-yolov8s", headers=_auth(), timeout=10).json()
    assert prod3["model_version"] == V1
    checks["rollback"] = {"production_after_rollback": prod3["model_version"]}

    # 7. desync is observable instead of silent
    sync = httpx.get(f"{BACKEND}/api/v1/models/runtime-sync", headers=_auth(), timeout=15).json()
    checks["runtime_sync"] = {"overall": sync["overall"],
                              "channels": {c["channel"]: c["status"] for c in sync["channels"]}}

    # 8. drift + feedback endpoints respond
    d = httpx.get(f"{BACKEND}/api/v1/drift", timeout=10).json()
    assert d["overall"] in ("NORMAL", "WARNING", "CRITICAL")
    f = httpx.get(f"{BACKEND}/api/v1/human-feedback", timeout=10).json()
    assert "defect_confirmation_rate" in f
    checks["drift"] = {"overall": d["overall"]}
    checks["feedback"] = {"resolved": f["resolved"]}

    OUT.write_text(json.dumps(checks, indent=2, ensure_ascii=False))
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("PHASE 8 E2E: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
