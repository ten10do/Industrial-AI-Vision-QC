"""Phase 6: UNKNOWN_ANOMALY review integration (6F/6G)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from vision_contract import AnomalyRegion, AnomalyResult, VisionResult, utc_now_iso

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ReviewCorrection, ReviewDecision, ReviewTask  # noqa: E402

from test_inspection_api import SAMPLE_JPG, StubInference  # noqa: E402
from test_reviews import _task_for_inspection  # noqa: E402


def _vision_result_unknown_anomaly() -> VisionResult:
    return VisionResult(
        inspection_id="insp-anom",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[],
        anomaly=AnomalyResult(
            model_name="patchcore-wrn50-2",
            model_version="mvtec-bottle-baseline",
            anomaly_score=0.42,
            threshold=0.2,
            is_anomalous=True,
            latency_ms=5.0,
            anomaly_map_png=None,
            regions=[
                AnomalyRegion(
                    bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
                    bbox_normalized=(0.05, 0.05, 0.3, 0.3),
                    area_ratio=0.0625,
                    region_score=0.55,
                )
            ],
        ),
        fusion_class="UNKNOWN_ANOMALY",
        latency_yolo_ms=10.0,
        latency_anomaly_ms=5.0,
        latency_fusion_ms=0.1,
        inference_latency_ms=15.1,
        device="cuda:0",
        timestamp=utc_now_iso(),
    )


@pytest.mark.asyncio
async def test_unknown_anomaly_creates_review_task_with_anomaly_snapshot(client, db_session, stub_infer):
    stub_infer(StubInference(result=_vision_result_unknown_anomaly()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-ANOM-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["quality_result"] == "REVIEW"
    assert body["fusion_class"] == "UNKNOWN_ANOMALY"
    assert body["is_anomalous"] is True
    assert body["anomaly_score"] == 0.42

    task = await _task_for_inspection(db_session, body["inspection_id"])
    assert task is not None
    assert task.is_anomalous is True
    assert task.anomaly_score == 0.42
    assert task.anomaly_threshold == 0.2
    assert task.anomaly_regions == [
        {"bbox_xyxy": [10.0, 10.0, 60.0, 60.0], "bbox_normalized": [0.05, 0.05, 0.3, 0.3], "area_ratio": 0.0625, "region_score": 0.55}
    ]


@pytest.mark.asyncio
async def test_resolve_unknown_anomaly_confirm_defect(client, db_session, stub_infer, auth):
    stub_infer(StubInference(result=_vision_result_unknown_anomaly()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-ANOM-2", "production_line": "line-a", "station": "qc-01"},
    )
    inspection_id = resp.json()["inspection_id"]
    task = await _task_for_inspection(db_session, inspection_id)
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"),
        json={"human_decision": "CONFIRM_DEFECT", "human_label": "new_crack", "reason": "verified under light"},
    )
    assert r.status_code == 200
    assert r.json()["decision"]["final_quality_result"] == "FAIL"
    assert r.json()["decision"]["human_label"] == "new_crack"
    assert r.json()["anomaly_score"] == 0.42

    candidates = (await client.get("/api/v1/training-candidates", headers=auth("release-manager"), params={"kind": "all"})).json()
    hit = next((c for c in candidates if c["inspection_id"] == inspection_id), None)
    assert hit is not None
    assert hit["human_label"] == "new_crack"
    assert hit["anomaly_score"] == 0.42
    assert hit["ai_label"] is None  # human label is new knowledge from the unknown anomaly
