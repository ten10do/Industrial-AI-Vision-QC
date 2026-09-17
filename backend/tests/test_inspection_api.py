from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from vision_contract import Detection, InferenceResult, utc_now_iso

from app.enums import QualityResult, Severity
from app.config import get_settings
from app.inference.client import (
    InferenceConnectionError,
    InferenceContractError,
    InferenceHTTPError,
    InferenceTimeoutError,
)
from app.models import Inspection, Product, QualityRule

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_IMAGE = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg"
SAMPLE_JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def contract(*, detections: list[dict] | None = None) -> dict:
    return InferenceResult(
        inspection_id="insp-fake",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[Detection(**d) for d in (detections or [])],
        inference_latency_ms=11.2,
        device="cuda:0",
        timestamp=utc_now_iso(),
    )


def detection(class_name: str, confidence: float, area_ratio: float) -> dict:
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
        "bbox_normalized": (0.05, 0.05, 0.25, 0.25),
        "defect_area_px": 1600.0,
        "defect_area_ratio": area_ratio,
    }


class StubInference:
    def __init__(self, *, result: InferenceResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error

    async def infer(self, image_bytes: bytes, filename: str = "image.jpg", request_id: str | None = None):
        if self.error is not None:
            raise self.error
        return self.result


async def count_inspections(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Inspection))
    return result.scalar_one()


async def test_create_inspection_pass_path(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("clean.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0001", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["quality_result"] == "PASS"
    assert body["status"] == "completed"
    assert body["defects"] == []
    assert await count_inspections(db_session) == 1


async def test_create_inspection_fail_path_with_seeded_rule(client, db_session, stub_infer):
    db_session.add(
        QualityRule(
            defect_type="crazing",
            min_confidence=0.5,
            max_area_ratio=1.0,
            action=QualityResult.FAIL,
            severity=Severity.HIGH,
            priority=1,
        )
    )
    await db_session.commit()
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.9, 0.1)])))

    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("crazing.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0002"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["quality_result"] == "FAIL"
    assert body["severity"] == "high"
    assert len(body["defects"]) == 1
    assert body["defects"][0]["class_name"] == "crazing"


async def test_fetch_inspection_and_product_history(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract(detections=[detection("scratches", 0.8, 0.05)])))
    r1 = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0003"},
    )
    r2 = await client.post(
        "/api/v1/inspections",
        files={"file": ("b.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0003"},
    )
    assert r1.status_code == r2.status_code == 201
    insp1, insp2 = r1.json()["inspection_id"], r2.json()["inspection_id"]

    fetched = await client.get(f"/api/v1/inspections/{insp1}")
    assert fetched.status_code == 200
    assert fetched.json()["product_id"] == "NEU-0003"

    history = await client.get("/api/v1/products/NEU-0003/inspections")
    assert history.status_code == 200
    assert [i["inspection_id"] for i in history.json()] == [insp1, insp2]


async def test_inference_timeout_maps_to_504(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceTimeoutError("timed out")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0004"},
    )
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "inference_failed"
    row = await db_session.execute(select(Inspection))
    assert row.scalar_one().status.value == "failed"


async def test_inference_http_500_maps_to_502(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceHTTPError(500, "boom")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0005"},
    )
    assert resp.status_code == 502


async def test_invalid_contract_maps_to_502(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceContractError("bad schema")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0006"},
    )
    assert resp.status_code == 502


async def test_connection_error_maps_to_504(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceConnectionError("unreachable")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0007"},
    )
    assert resp.status_code == 504


async def test_invalid_image_rejected_422(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("fake.png", b"this is not an image", "image/png")},
        data={"product_id": "NEU-0008"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_image"
    assert await count_inspections(db_session) == 0


@pytest.mark.parametrize("params", [
    {"quality_result": "UNKNOWN"},
    {"status": "not-a-status"},
    {"date_from": "not-a-date"},
    {"date_to": "2026-99-99"},
])
async def test_invalid_inspection_filters_return_422(client, params):
    response = await client.get("/api/v1/inspections", params=params)
    assert response.status_code == 422


async def test_oversized_image_rejected_413_before_inference(
    client, db_session, stub_infer, monkeypatch
):
    class UnexpectedInference:
        async def infer(self, *args, **kwargs):
            raise AssertionError("inference must not run for an oversized upload")

    stub_infer(UnexpectedInference())
    monkeypatch.setattr(get_settings(), "max_upload_bytes", len(SAMPLE_JPG))

    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("too-large.jpg", SAMPLE_JPG + b"x", "image/jpeg")},
        data={"product_id": "NEU-OVERSIZED"},
    )

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
    assert await count_inspections(db_session) == 0


async def test_idempotency_key_returns_same_inspection(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    payload = {
        "files": {"file": ("a.png", SAMPLE_JPG, "image/png")},
        "data": {"product_id": "NEU-0009", "idempotency_key": "key-1"},
    }
    first = await client.post("/api/v1/inspections", **payload)
    second = await client.post("/api/v1/inspections", **payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["inspection_id"] == second.json()["inspection_id"]
    assert await count_inspections(db_session) == 1


async def test_db_rollback_on_commit_failure(client, db_session, stub_infer, monkeypatch):
    stub_infer(StubInference(result=contract()))

    original_commit = db_session.commit

    async def failing_commit():
        raise SQLAlchemyError("simulated db failure")

    monkeypatch.setattr(db_session, "commit", failing_commit)
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0010"},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "db_write_failed"
    assert await count_inspections(db_session) == 0
    monkeypatch.setattr(db_session, "commit", original_commit)


async def test_storage_failure_fails_closed(client, db_session, stub_infer, monkeypatch):
    """P0: a failed raw-image write must halt the inspection. The row lands in
    FAILED with no quality verdict, no image provenance, and the API answers
    500 image_storage_failed (never PASS/RELEASE)."""
    stub_infer(StubInference(result=contract()))

    def disk_full(inspection_id: str, data: bytes):
        raise OSError("disk full")

    from app.storage import get_storage

    monkeypatch.setattr(get_storage(), "save", disk_full)
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("clean.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0011"},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "image_storage_failed"

    row = (await db_session.execute(select(Inspection))).scalar_one()
    assert row.status.value == "failed"
    assert "image storage failed" in row.error_message
    # fail-closed: no quality verdict, no release, no provenance to point at
    assert row.quality_result is None
    assert row.final_quality_result is None
    assert row.image_path is None
    assert row.image_sha256 is None
    assert row.image_media_type is None


async def test_success_path_records_real_artifact_metadata(client, db_session, stub_infer):
    """P0: image_path must be the real stored URI (never the upload filename),
    and the sha256/media_type are server-side detected, not client-claimed."""
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("clean.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0012"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    expected_digest = hashlib.sha256(SAMPLE_JPG).hexdigest()
    # magic bytes 0xffd8ff -> jpeg even though the client claimed image/png
    assert body["image_sha256"] == expected_digest
    assert body["image_media_type"] == "image/jpeg"

    row = (await db_session.execute(select(Inspection))).scalar_one()
    assert row.image_sha256 == expected_digest
    assert row.image_media_type == "image/jpeg"
    assert row.image_path is not None
    assert not row.image_path.endswith("clean.png")
    target = Path(row.image_path) if Path(row.image_path).is_absolute() else PROJECT_ROOT / row.image_path
    assert target.is_file(), f"recorded image URI does not exist on disk: {row.image_path}"
