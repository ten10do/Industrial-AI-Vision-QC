from __future__ import annotations

import asyncio
import csv
import io
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enums import HumanDecision, QualityResult, ReviewTaskStatus  # noqa: E402
from app.models import Inspection, ReviewCorrection, ReviewDecision, ReviewTask  # noqa: E402

from test_inspection_api import SAMPLE_JPG, StubInference, contract, detection  # noqa: E402

REVIEW = {"action": "REVIEW", "severity": "medium"}
FAIL_RULE = {"action": "FAIL", "severity": "high"}
PASS_RULE = {"action": "PASS", "severity": "low"}


@pytest.fixture(autouse=True)
async def _reset_rt_metrics():
    """The realtime metrics singleton is process-global; reset between tests."""
    from app.metrics import metrics as rt

    await rt.reset()
    yield
    await rt.reset()


async def _post(client, *, detections=None, stub=None, product_id="P-REV-1"):
    from test_inspection_api import contract as mk_contract

    stub = stub or StubInference(result=mk_contract(detections=detections or []))
    client.app.dependency_overrides.clear()
    from app.api.inspections import get_inspection_service
    from app.services.inspection_service import InspectionService

    client.app.dependency_overrides[get_inspection_service] = lambda: InspectionService(inference_client=stub)
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": product_id, "production_line": "line-a", "station": "qc-01", "batch_id": "b1"},
    )
    return resp


async def _seed_rules(session: AsyncSession):
    from app.models import QualityRule

    session.add_all(
        [
            QualityRule(defect_type="crazing", min_confidence=0.3, max_area_ratio=1.0, action=QualityResult.REVIEW, severity="medium", priority=10, rule_version=1),
            QualityRule(defect_type="scratches", min_confidence=0.9, max_area_ratio=1.0, action=QualityResult.FAIL, severity="high", priority=20, rule_version=1),
        ]
    )
    await session.commit()


async def _task_for_inspection(session: AsyncSession, inspection_id: str) -> ReviewTask | None:
    result = await session.execute(
        select(ReviewTask)
        .join(Inspection, Inspection.id == ReviewTask.inspection_id)
        .options(
            selectinload(ReviewTask.decision).selectinload(ReviewDecision.corrections),
        )
        .where(Inspection.inspection_id == inspection_id)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


@pytest.mark.asyncio
async def test_review_creates_task(client, db_session, stub_infer):
    """5B: REVIEW inspection auto-creates a PENDING task with a frozen AI snapshot."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-1", "production_line": "line-a", "station": "qc-01", "batch_id": "b1"},
    )
    assert resp.status_code == 201
    inspection_id = resp.json()["inspection_id"]
    assert resp.json()["quality_result"] == "REVIEW"
    assert resp.json()["final_quality_result"] is None

    task = await _task_for_inspection(db_session, inspection_id)
    assert task is not None
    assert task.status == ReviewTaskStatus.PENDING
    assert task.ai_quality_result == "REVIEW"
    assert task.ai_defects_snapshot[0]["class_name"] == "crazing"
    assert task.ai_defects_snapshot[0]["confidence"] == 0.42
    assert task.product_id == "P-REV-1"
    assert task.image_url == f"/api/v1/inspections/{inspection_id}/image"


@pytest.mark.asyncio
async def test_pass_and_fail_do_not_create_task(client, db_session, stub_infer):
    """5B: PASS and FAIL inspections never enter the review queue."""
    await _seed_rules(db_session)
    # PASS: no detections -> engine yields PASS
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-PASS-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    assert resp.json()["quality_result"] == "PASS"
    assert resp.json()["final_quality_result"] == "PASS"
    assert await _task_for_inspection(db_session, resp.json()["inspection_id"]) is None

    stub_infer(StubInference(result=contract(detections=[detection("scratches", 0.95, 0.2)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-FAIL-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    assert resp.json()["quality_result"] == "FAIL"
    assert resp.json()["final_quality_result"] == "FAIL"
    assert await _task_for_inspection(db_session, resp.json()["inspection_id"]) is None


@pytest.mark.asyncio
async def test_system_failed_does_not_create_task(client, db_session, stub_infer):
    """5B: process FAILED (inference down) never creates a review task."""
    from app.inference.client import InferenceTimeoutError

    stub_infer(StubInference(error=InferenceTimeoutError("timed out")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-SYS-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 504
    tasks = await db_session.execute(select(func.count()).select_from(ReviewTask))
    assert tasks.scalar_one() == 0


@pytest.mark.asyncio
async def test_duplicate_task_prevention(client, db_session, stub_infer):
    """5B: idempotent replay of the same inspection yields at most one task."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    data = {
        "files": {"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        "data": {"product_id": "P-REV-2", "production_line": "line-a", "station": "qc-01", "idempotency_key": "k-1"},
    }
    r1 = await client.post("/api/v1/inspections", **data)
    r2 = await client.post("/api/v1/inspections", **data)
    assert r1.status_code == 201 and r2.status_code == 200
    count = await db_session.execute(
        select(func.count()).select_from(ReviewTask).join(Inspection).where(
            Inspection.inspection_id == r1.json()["inspection_id"]
        )
    )
    assert count.scalar_one() == 1


@pytest_asyncio.fixture
async def concurrent_env(tmp_path):
    """Dedicated client whose requests get SEPARATE sessions on a shared
    file-based sqlite. Two concurrent claims race at the DB level, which is
    the exact concurrency the 5D gate requires."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.database import get_session as gs
    from app.main import create_app
    from app.models import Base

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'conc.db'}",
        connect_args={"timeout": 30},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override():
        async with factory() as s:
            yield s

    application = create_app()
    application.dependency_overrides[gs] = override
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, application, factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_claim_and_concurrent_claim(concurrent_env, tmp_path, auth):
    """5D: exactly one of two concurrent claims succeeds; the loser gets 409."""
    client, application, factory = concurrent_env

    from app.api.inspections import get_inspection_service
    from app.models import QualityRule
    from app.services.inspection_service import InspectionService

    async with factory() as s:
        s.add_all(
            [
                QualityRule(defect_type="crazing", min_confidence=0.3, max_area_ratio=1.0, action=QualityResult.REVIEW, severity="medium", priority=10, rule_version=1),
            ]
        )
        await s.commit()

    application.dependency_overrides[get_inspection_service] = lambda: InspectionService(
        inference_client=StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)]))
    )
    resp = await client.post(
        "/api/v1/inspections",
        headers=auth("operator"),
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-3", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201

    async with factory() as s:
        task = (await s.execute(select(ReviewTask))).scalars().first()
    assert task is not None

    results = await asyncio.gather(
        client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={}),
        client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_b"), json={}),
        return_exceptions=False,
    )
    codes = sorted(r.status_code for r in results)
    assert codes == [200, 409]
    winner = next(r for r in results if r.status_code == 200).json()
    loser = next(r for r in results if r.status_code == 409).json()
    assert winner["status"] == "IN_REVIEW"
    assert winner["assigned_to"] in ("tester-reviewer-a", "tester-reviewer-b")
    assert loser["error"]["code"] == "already_claimed"
    assert winner["version"] == 2  # optimistic lock bumped


@pytest.mark.asyncio
async def test_resolve_requires_claim_and_owner(client, db_session, stub_infer, auth):
    """5E/5F: resolving without claim or by a non-owner is rejected."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-4", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])

    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "PASS"},
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_claimed"

    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_b"), json={"human_decision": "PASS"},
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_owner"


@pytest.mark.asyncio
async def test_resolve_pass_override(client, db_session, stub_infer, auth):
    """5E: human PASS overrides AI REVIEW -> final PASS, AI judgment preserved."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-5", "production_line": "line-a", "station": "qc-01"},
    )
    inspection_id = resp.json()["inspection_id"]
    task = await _task_for_inspection(db_session, inspection_id)
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "PASS", "reason": "false positive"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "RESOLVED"
    assert body["decision"]["human_decision"] == "PASS"
    assert body["decision"]["final_quality_result"] == "PASS"
    assert body["decision"]["ai_quality_result"] == "REVIEW"

    insp = await db_session.execute(
        select(Inspection).where(Inspection.inspection_id == inspection_id)
    )
    inspection = insp.scalar_one()
    # AI 原始判断不可变；final 为人工结果（5G）
    assert inspection.quality_result == QualityResult.REVIEW
    assert inspection.final_quality_result == QualityResult.PASS


@pytest.mark.asyncio
async def test_resolve_confirm_defect(client, db_session, stub_infer, auth):
    """5E: confirm AI defect -> final FAIL."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-6", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CONFIRM_DEFECT", "human_label": "crazing", "reason": "visible defect"},
    )
    assert r.status_code == 200
    assert r.json()["decision"]["final_quality_result"] == "FAIL"
    assert r.json()["decision"]["human_label"] == "crazing"


@pytest.mark.asyncio
async def test_resolve_correct_label_and_validation(client, db_session, stub_infer, auth):
    """5E: correct label -> final FAIL; confirm without a label -> 422."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-7", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})

    bad = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CONFIRM_DEFECT"},
    )
    assert bad.status_code == 422

    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CORRECT_DEFECT", "human_label": "scratches", "reason": "it is scratches"},
    )
    assert r.status_code == 200
    assert r.json()["decision"]["final_quality_result"] == "FAIL"
    assert r.json()["decision"]["human_label"] == "scratches"
    assert r.json()["decision"]["ai_defects_snapshot"][0]["class_name"] == "crazing"  # AI snapshot frozen


@pytest.mark.asyncio
async def test_resolve_already_resolved_conflict(client, db_session, stub_infer, auth):
    """5F: resolving twice conflicts; second attempt gets 409."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-8", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    ok = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "PASS"},
    )
    assert ok.status_code == 200
    again = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CONFIRM_DEFECT", "human_label": "crazing"},
    )
    assert again.status_code == 409 and again.json()["error"]["code"] == "already_resolved"
    # 原决策未被覆盖
    decisions = await db_session.execute(select(func.count()).select_from(ReviewDecision))
    assert decisions.scalar_one() == 1


@pytest.mark.asyncio
async def test_correction_appends_without_overwrite(client, db_session, stub_infer, auth):
    """5F: post-resolve corrections append an audit record; the original
    decision stays untouched."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-9", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "PASS", "reason": "first pass"},
    )
    corr = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/corrections",
        headers=auth("reviewer_b"), json={"field_changed": "human_decision", "new_value": {"value": "CONFIRM_DEFECT"}, "reason": "re-inspection found the defect"},
    )
    assert corr.status_code == 200

    task2 = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    assert task2.status == ReviewTaskStatus.RESOLVED
    # 原 human_decision 保持 PASS（不可静默覆盖）
    assert task2.decision.human_decision == HumanDecision.PASS
    assert len(task2.decision.corrections) == 1
    assert task2.decision.corrections[0].field_changed == "human_decision"
    assert task2.decision.corrections[0].old_value == {"value": "PASS"}


@pytest.mark.asyncio
async def test_review_list_filters(client, db_session, stub_infer, auth):
    """5C: queue list supports status/line/station/batch filters."""
    await _seed_rules(db_session)
    for i in range(2):
        stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
        await client.post(
            "/api/v1/inspections",
            files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
            data={"product_id": f"P-REV-{i}", "production_line": f"line-{i}", "station": "qc-01", "batch_id": f"batch-{i}"},
        )
    r = await client.get("/api/v1/reviews", headers=auth("reviewer_a"), params={"status": "PENDING"})
    assert r.status_code == 200
    assert len(r.json()) == 2
    r = await client.get("/api/v1/reviews", headers=auth("reviewer_a"), params={"production_line": "line-0"})
    assert len(r.json()) == 1 and r.json()[0]["product_id"] == "P-REV-0"
    r = await client.get("/api/v1/reviews", headers=auth("reviewer_a"), params={"batch_id": "batch-1"})
    assert len(r.json()) == 1


@pytest.mark.asyncio
async def test_training_candidates_export(client, db_session, stub_infer, auth):
    """5J: resolved reviews export as JSON/CSV with AI + human labels."""
    await _seed_rules(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-REV-TR", "production_line": "line-a", "station": "qc-01"},
    )
    task = await _task_for_inspection(db_session, resp.json()["inspection_id"])
    await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CORRECT_DEFECT", "human_label": "scratches", "reason": "wrong class"},
    )

    js = await client.get("/api/v1/training-candidates", headers=auth("release-manager"), params={"kind": "corrected"})
    assert js.status_code == 200
    candidates = js.json()
    assert len(candidates) == 1
    assert candidates[0]["ai_label"] == "crazing"
    assert candidates[0]["human_label"] == "scratches"
    assert candidates[0]["ai_confidence"] == 0.42
    assert candidates[0]["agreement"] is False
    # semantic fix (Phase 9): three distinct source identities
    assert candidates[0]["source_model_version"] == "phase1-baseline"
    assert candidates[0]["source_deployment_version"] == "2026.08.1"
    assert candidates[0]["source_dataset_version"] is not None
    assert candidates[0]["image_url"].endswith("/image")

    csv_resp = await client.get("/api/v1/training-candidates", headers=auth("release-manager"), params={"kind": "all", "format": "csv"})
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    rows = list(csv.DictReader(io.StringIO(csv_resp.text)))
    assert len(rows) == 1 and rows[0]["human_label"] == "scratches"


@pytest.mark.asyncio
async def test_review_metrics_semantics(client, db_session, stub_infer, auth):
    """5K: review metrics use the documented semantics."""
    await _seed_rules(db_session)
    # 1 REVIEW (resolved with CONFIRM) + 1 PASS (no task)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-M-1", "production_line": "line-a", "station": "qc-01"},
    )
    stub_infer(StubInference(result=contract()))  # no detections -> PASS, no task
    await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-M-2", "production_line": "line-a", "station": "qc-01"},
    )
    tasks = await db_session.execute(select(ReviewTask))
    task = tasks.scalars().first()
    assert task is not None
    claim_resp = await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    assert claim_resp.status_code == 200, claim_resp.text
    resolve_resp = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/resolve",
        headers=auth("reviewer_a"), json={"human_decision": "CONFIRM_DEFECT", "human_label": "crazing"},
    )
    assert resolve_resp.status_code == 200, resolve_resp.text

    m = (await client.get("/api/v1/reviews-metrics", headers=auth("reviewer_a"))).json()
    assert m["pending_review_count"] == 0
    assert m["resolved"] == 1
    assert m["review_rate"] == 0.5  # 1 REVIEW / 2 completed
    assert m["defect_confirmation_rate"] == 1.0  # CONFIRM confirms AI
    assert m["ai_human_label_agreement_rate"] == 1.0  # label == AI top defect
    assert m["override_rate"] == 0.0
    assert m["corrected_label_count"] == 0
    assert m["average_review_wait_time_s"] is not None


@pytest.mark.asyncio
async def test_invalid_review_status_filter_returns_422(client, auth):
    response = await client.get(
        "/api/v1/reviews", params={"status": "UNKNOWN"}, headers=auth("reviewer_a")
    )
    assert response.status_code == 422
