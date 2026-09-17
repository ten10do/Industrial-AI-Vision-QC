"""Phase 8: model registry, promotion gate, drift, deployment manifest."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mlops.drift import classify_ks, classify_psi, defect_distribution_delta, ks_statistic, psi, review_rate_delta
from app.mlops.manifest import load_manifest, reset_manifest_cache, sha256_of, validate_artifacts
from app.mlops.promotion_gate import evaluate
from app.models import Base, ModelRegistry
from app.security.auth import ROLE_ADMIN, ROLE_APPROVER, ROLE_PIPELINE, Principal
from app.services.registry_service import RegistryError, RegistryService, provenance_for

PIPELINE = Principal(subject="eval-pipeline", roles=frozenset({ROLE_PIPELINE}))
APPROVER = Principal(subject="release-manager", roles=frozenset({ROLE_APPROVER}))
APPROVER_NAME = "qa-director"
REASON = "unit test promotion acceptance"

ARTIFACT_URI = "inference-service/models/best.pt"


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture(scope="session")
def artifact_sha256(artifact) -> str:
    """The sha256 of the file registered as ARTIFACT_URI.

    Depends on the shared ``artifact`` fixture so the file is guaranteed to
    exist before it is hashed: on a fresh checkout (CI) there is no real
    model, and the fixture synthesizes a stand-in at exactly this path, so
    the hash describes the same bytes the server will re-hash."""
    return sha256_of(Path(__file__).resolve().parents[2] / ARTIFACT_URI)


def _gate(entry: ModelRegistry) -> object:
    """Evaluate the gate the way the API does: policy thresholds plus the
    provenance recorded on the row."""
    return evaluate(
        entry.model_type,
        metrics=entry.metadata_json or {},
        domain_validated=entry.domain_validated,
        required_domain="steel",
        provenance=provenance_for(entry),
    )


def _evidence(eval_report: dict) -> dict:
    return {
        "domain": "steel",
        "dataset_version": "neu-det-yolo-v1",
        "eval_report_uri": eval_report["uri"],
        "eval_report_sha256": eval_report["sha256"],
        "validated_by": "eval-pipeline",
    }


# ---- promotion gate ----

FULL_PROVENANCE = {
    "metrics_attested": True,
    "artifact_hash_verified": True,
    "domain_evidence_verified": True,
    "domain": "steel",
}


def test_yolo_gate_pass():
    g = evaluate(
        "yolo",
        metrics={"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
        domain_validated=True, required_domain="steel", provenance=FULL_PROVENANCE,
    )
    assert g.passed is True
    assert g.blocked == []


def test_yolo_gate_reject_low_metric():
    g = evaluate(
        "yolo",
        metrics={"mAP50": 0.4, "recall": 0.78, "latency_p95_ms": 21.5},
        domain_validated=True, required_domain="steel", provenance=FULL_PROVENANCE,
    )
    assert g.passed is False
    assert any("mAP50" in b for b in g.blocked)


def test_yolo_gate_reject_excessive_latency():
    """Acceptance audit (minimal addition, disclosed): the gate code already
    rejects latency_p95 above threshold; this pins it with a dedicated test."""
    g = evaluate(
        "yolo",
        metrics={"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 5000.0},
        domain_validated=True, required_domain="steel", provenance=FULL_PROVENANCE,
    )
    assert g.passed is False
    assert any("latency" in b for b in g.blocked)


def test_patchcore_domain_mismatch_rejects_perfect_auroc():
    """The MVTec baseline has image_auroc=1.0 but steel_domain_validated=false:
    it must NEVER be promoted as a steel production model (8F)."""
    g = evaluate(
        "patchcore",
        metrics={"image_auroc": 1.0, "pixel_auroc": 0.986, "latency_ms": 755.0},
        domain_validated=False, required_domain="steel",
        provenance={"metrics_attested": True, "artifact_hash_verified": True,
                    "domain_evidence_verified": False, "domain": "steel"},
    )
    assert g.passed is False
    assert any("domain" in b for b in g.blocked)


# ---- registry ----

async def _register_yolo(svc, session, eval_report, artifact_sha256, version="1.0.0", domain=True, metrics=None):
    return await svc.register(
        session, actor=PIPELINE, model_name="neu-yolov8s", model_version=version, model_type="yolo",
        artifact_uri=ARTIFACT_URI, artifact_sha256=artifact_sha256,
        dataset_version="neu-det-yolo-v1", training_run_id=f"run-{version}",
        metrics=metrics or {"mAP50": 0.82, "recall": 0.78, "latency_p95_ms": 21.5},
        domain_validated=domain, domain_evidence=_evidence(eval_report) if domain else None,
    )


async def _promote(svc, session, entry):
    return await svc.promote(
        session, entry, gate=_gate(entry), required_domain="steel",
        actor=APPROVER, approved_by=APPROVER_NAME, reason=REASON,
    )


@pytest.mark.asyncio
async def test_register_and_duplicate_version(db, eval_report, artifact_sha256):
    svc = RegistryService()
    m = await _register_yolo(svc, db, eval_report, artifact_sha256)
    assert m.status == "CANDIDATE"
    await db.commit()
    with pytest.raises(RegistryError) as e:
        await _register_yolo(svc, db, eval_report, artifact_sha256, version="1.0.0")
    assert e.value.code == "duplicate_version"


@pytest.mark.asyncio
async def test_register_verifies_the_artifact_hash(db, eval_report, artifact_sha256):
    """A wrong hash is stored, but flagged unverified, so the gate still
    blocks the promotion."""
    svc = RegistryService()
    m = await _register_yolo(svc, db, eval_report, "0" * 64)
    await db.commit()
    assert m.artifact_hash_verified is False
    assert _gate(m).passed is False


@pytest.mark.asyncio
async def test_production_unique_after_promote(db, eval_report, artifact_sha256):
    svc = RegistryService()
    v1 = await _register_yolo(svc, db, eval_report, artifact_sha256, version="1.0.0")
    await db.commit()
    assert _gate(v1).passed is True, _gate(v1).blocked
    await _promote(svc, db, v1)
    await db.commit()

    v2 = await _register_yolo(svc, db, eval_report, artifact_sha256, version="2.0.0",
                              metrics={"mAP50": 0.9, "recall": 0.85, "latency_p95_ms": 18.0})
    await db.commit()
    await _promote(svc, db, v2)
    await db.commit()

    # only v2 is PRODUCTION; v1 was archived
    prod = await svc.get_production(db, "neu-yolov8s")
    assert prod is not None and prod.model_version == "2.0.0"
    v1r = await svc.get_by_version(db, "neu-yolov8s", "1.0.0")
    assert v1r.status == "ARCHIVED"


@pytest.mark.asyncio
async def test_promote_rejected_by_domain(db, eval_report, artifact_sha256):
    svc = RegistryService()
    bad = await svc.register(
        session=db, actor=PIPELINE, model_name="mvtec-patchcore", model_version="1.0.0",
        model_type="patchcore",
        artifact_uri="inference-service/models/patchcore-bottle/bank.npz", artifact_sha256="x",
        dataset_version="mvtec-bottle-v1", metrics={"image_auroc": 1.0, "pixel_auroc": 0.986, "latency_ms": 755.0},
        domain_validated=False,  # MVTec baseline NOT validated for steel
    )
    await db.commit()
    with pytest.raises(RegistryError) as e:
        await _promote(svc, db, bad)
    assert e.value.code == "promotion_gate_failed"


@pytest.mark.asyncio
async def test_promote_requires_a_distinct_approver(db, eval_report, artifact_sha256):
    svc = RegistryService()
    v = await _register_yolo(svc, db, eval_report, artifact_sha256)
    await db.commit()
    with pytest.raises(RegistryError) as e:
        await svc.promote(db, v, gate=_gate(v), required_domain="steel", actor=APPROVER,
                          approved_by=APPROVER.subject, reason=REASON)
    assert e.value.code == "self_approval_forbidden"


@pytest.mark.asyncio
async def test_rollback_switches_production(db, eval_report, artifact_sha256):
    svc = RegistryService()
    v1 = await _register_yolo(svc, db, eval_report, artifact_sha256, version="1.0.0")
    v2 = await _register_yolo(svc, db, eval_report, artifact_sha256, version="2.0.0")
    await db.commit()
    for v in (v1, v2):
        await _promote(svc, db, v)
        await db.commit()

    assert (await svc.get_production(db, "neu-yolov8s")).model_version == "2.0.0"
    await svc.rollback(db, "neu-yolov8s", "1.0.0", actor=APPROVER,
                       approved_by=APPROVER_NAME, reason="rollback unit test")
    await db.commit()
    assert (await svc.get_production(db, "neu-yolov8s")).model_version == "1.0.0"


@pytest.mark.asyncio
async def test_audit_trail_records_every_transition(db, eval_report, artifact_sha256):
    svc = RegistryService()
    v = await _register_yolo(svc, db, eval_report, artifact_sha256)
    await db.commit()
    await _promote(svc, db, v)
    await db.commit()
    trail = await svc.audit_trail(db, v.id)
    assert [r.action for r in trail] == ["register", "attest", "promote"]
    assert all(r.outcome == "APPLIED" for r in trail)
    assert trail[-1].approved_by == APPROVER_NAME
    assert trail[-1].gate["passed"] is True


# ---- drift ----

def test_psi_normal_vs_warning():
    baseline = [0.2] * 40 + [0.8] * 60
    same = [0.2] * 45 + [0.8] * 55
    assert psi(baseline, same) < 0.1
    shifted = [0.8] * 90 + [0.2] * 10
    assert classify_psi(psi(baseline, shifted)) in ("WARNING", "CRITICAL")


def test_ks_classify():
    a = [0.1] * 50 + [0.9] * 50
    b = [0.1] * 50 + [0.9] * 50
    assert ks_statistic(a, b) < 0.01
    # a clearly separated distribution (same support, very different CDF)
    c = [0.5] * 100
    assert classify_ks(ks_statistic(a, c)) == "CRITICAL"


def test_review_rate_delta_and_defect_delta():
    assert review_rate_delta(0.2, 0.22) == "NORMAL"
    assert review_rate_delta(0.2, 0.6) == "CRITICAL"
    assert defect_distribution_delta({"crazing": 0.5, "inclusion": 0.5}, {"crazing": 0.5, "inclusion": 0.5}) == "NORMAL"
    assert defect_distribution_delta({"crazing": 0.5, "inclusion": 0.5}, {"crazing": 0.95, "inclusion": 0.05}) == "CRITICAL"


# ---- deployment manifest ----

def test_manifest_load_and_artifacts():
    reset_manifest_cache()
    m = load_manifest()
    assert m["vision_stack_version"] == "2026.08.1"
    assert m["yolo"]["model"] == "neu-yolov8s"
    assert m["patchcore"]["steel_domain_validated"] is False
    assert m["fusion"]["version"] == "1.0"


@pytest.mark.artifact
def test_manifest_artifact_sha256_matches_files():
    """The manifest pins real artifact hashes; the deployed files must match
    (8E: a wrong hash must make the stack not-ready, not silently load)."""
    reset_manifest_cache()
    m = load_manifest()
    root = Path(__file__).resolve().parents[2]
    problems = validate_artifacts(m, root)
    assert problems == [], problems
    # and the recorded hashes equal the real files
    assert sha256_of(root / "inference-service/models/best.pt") == m["yolo"]["sha256"]
    assert sha256_of(root / "inference-service/models/patchcore-bottle/bank.npz") == m["patchcore"]["sha256"]


def test_manifest_artifact_wrong_hash_detected(tmp_path):
    f = tmp_path / "fake.pt"
    f.write_bytes(b"garbage")
    m = {
        "vision_stack_version": "x",
        "yolo": {"model": "m", "version": "1", "artifact_uri": str(f), "sha256": "deadbeef"},
        "patchcore": {"model": "p", "version": "1", "artifact_uri": "nonexistent.npz", "sha256": "x"},
    }
    problems = validate_artifacts(m, tmp_path)
    assert any("sha256 mismatch" in p for p in problems)
    assert any("missing" in p for p in problems)
