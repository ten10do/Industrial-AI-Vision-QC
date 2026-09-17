"""Human-in-the-loop review closed loop (Phase 5).

Owns the review task lifecycle:

  REVIEW inspection persisted  -> task PENDING (idempotent, AI snapshot frozen)
  claim (PENDING -> IN_REVIEW)  -> conditional UPDATE; exactly one concurrent
                                   caller wins, the rest get 409
  resolve (IN_REVIEW -> RESOLVED) -> decision recorded (immutable), inspection
                                   final_quality_result set, audit correction
                                   supported afterwards via review_corrections

DB is the source of truth; WebSocket events are fire-and-forget notifications.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..enums import HumanDecision, QualityResult, ReviewTaskStatus
from ..events import ReviewEvent
from ..models import Inspection, ReviewCorrection, ReviewDecision, ReviewTask
from ..ws import schedule_broadcast

logger = logging.getLogger(__name__)

_SEVERITY_PRIORITY = {"critical": 100, "high": 200, "medium": 300, "low": 400}

FINAL_BY_DECISION = {
    HumanDecision.PASS: QualityResult.PASS,
    HumanDecision.CONFIRM_DEFECT: QualityResult.FAIL,
    HumanDecision.CORRECT_DEFECT: QualityResult.FAIL,
    HumanDecision.OTHER_DEFECT: QualityResult.FAIL,
}


class ReviewConflictError(Exception):
    """409: the task is in a conflicting state (already claimed / resolved)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ReviewValidationError(Exception):
    """422: invalid decision payload."""

    def __init__(self, message: str) -> None:
        self.code = "invalid_review"
        super().__init__(message)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot_defects(inspection: Inspection) -> list[dict]:
    """Frozen AI defect snapshot (5A): taken once at task creation and stored
    on both the task and the decision, never re-assembled from live data."""
    return [
        {
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox_xyxy": list(d.bbox_xyxy),
            "bbox_normalized": list(d.bbox_normalized),
            "defect_area_px": d.defect_area_px,
            "defect_area_ratio": d.defect_area_ratio,
            "severity": d.severity.value if d.severity else None,
            "matched_rule": d.matched_rule,
        }
        for d in inspection.defects
    ]


def _top_defect(snapshot: list[dict]) -> tuple[str | None, float | None]:
    best = max(snapshot, key=lambda d: d["confidence"], default=None)
    if best is None:
        return None, None
    return best["class_name"], best["confidence"]


def _review_event(task: ReviewTask, event_type: str, **extra) -> ReviewEvent:
    top_class, top_conf = _top_defect(task.ai_defects_snapshot)
    return ReviewEvent(
        event_type=event_type,  # type: ignore[arg-type]
        review_task_id=task.review_task_id,
        inspection_id=task.inspection.inspection_id if task.inspection else task.inspection_id,
        product_id=task.product_id,
        status=task.status.value,
        priority=task.priority,
        assigned_to=task.assigned_to,
        top_defect_class=top_class,
        top_confidence=top_conf,
        severity=task.ai_severity,
        model_version=task.ai_model_version,
        image_url=task.image_url,
        anomaly_score=task.anomaly_score,
        is_anomalous=task.is_anomalous,
        anomaly_map_url=task.anomaly_map_url,
        **extra,
    )


class ReviewService:
    async def create_task_for_inspection(self, session: AsyncSession, inspection: Inspection) -> ReviewTask | None:
        """5B: only completed REVIEW inspections get a review task; system
        FAILED never enters the queue. Idempotent: at most one task per
        inspection over its lifetime (global unique constraint, Phase 6
        pre-gate; re-review is not supported)."""
        if inspection.status.value != "completed" or inspection.quality_result != QualityResult.REVIEW:
            return None

        existing_row = await session.execute(
            select(ReviewTask).where(ReviewTask.inspection_id == inspection.id)
        )
        existing = existing_row.scalar_one_or_none()
        if existing is not None:
            return existing

        severity = inspection.severity.value if inspection.severity else None
        task = ReviewTask(
            review_task_id=f"rt-{uuid.uuid4().hex[:12]}",
            inspection_id=inspection.id,
            status=ReviewTaskStatus.PENDING,
            priority=_SEVERITY_PRIORITY.get(severity or "medium", 300),
            version=1,
            ai_quality_result=inspection.quality_result.value,
            ai_defects_snapshot=_snapshot_defects(inspection),
            ai_model_version=inspection.model_version,
            ai_rule_version=inspection.rule_version,
            ai_severity=severity,
            product_id=inspection.product.product_id,
            production_line=inspection.product.production_line,
            station=inspection.product.station,
            batch_id=inspection.batch_id,
            image_url=f"/api/v1/inspections/{inspection.inspection_id}/image",
            # Phase 6 anomaly snapshot for UNKNOWN_ANOMALY review (6G)
            anomaly_score=inspection.anomaly_score,
            anomaly_threshold=inspection.anomaly_threshold,
            is_anomalous=inspection.is_anomalous,
            anomaly_regions=inspection.anomaly_regions,
            anomaly_map_url=(
                f"/api/v1/inspections/{inspection.inspection_id}/anomaly-map"
                if inspection.is_anomalous and inspection.anomaly_map_path
                else None
            ),
        )
        task.inspection = inspection
        session.add(task)
        try:
            await session.flush()
            await session.commit()
        except Exception:
            # unique constraint race: another task exists for this inspection
            await session.rollback()
            return None
        schedule_broadcast(_review_event(task, "review.created").to_broadcast())
        return task

    async def list_tasks(
        self,
        session: AsyncSession,
        *,
        status: ReviewTaskStatus | None = None,
        priority: int | None = None,
        defect_type: str | None = None,
        production_line: str | None = None,
        station: str | None = None,
        batch_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ReviewTask]:
        stmt = select(ReviewTask).options(
            selectinload(ReviewTask.inspection).selectinload(Inspection.defects),
            selectinload(ReviewTask.inspection).selectinload(Inspection.product),
            selectinload(ReviewTask.decision),
        )
        if status:
            stmt = stmt.where(ReviewTask.status == status)
        if priority is not None:
            stmt = stmt.where(ReviewTask.priority <= priority)
        if defect_type:
            stmt = stmt.where(ReviewTask.ai_defects_snapshot.cast(str).contains(defect_type))
        if production_line:
            stmt = stmt.where(ReviewTask.production_line == production_line)
        if station:
            stmt = stmt.where(ReviewTask.station == station)
        if batch_id:
            stmt = stmt.where(ReviewTask.batch_id == batch_id)
        stmt = stmt.order_by(ReviewTask.priority.asc(), ReviewTask.created_at.asc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        return list(result.scalars())

    async def get_task(self, session: AsyncSession, task_id: str) -> ReviewTask | None:
        stmt = (
            select(ReviewTask)
            .options(
                selectinload(ReviewTask.inspection).selectinload(Inspection.defects),
                selectinload(ReviewTask.inspection).selectinload(Inspection.product),
                selectinload(ReviewTask.decision).selectinload(ReviewDecision.corrections),
            )
            .where(ReviewTask.review_task_id == task_id)
            .execution_options(populate_existing=True)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def claim(self, session: AsyncSession, task_id: str, reviewer: str) -> ReviewTask:
        """5D: conditional UPDATE PENDING -> IN_REVIEW. Only one concurrent
        claim can win; losers get ReviewConflictError(409)."""
        task = await self.get_task(session, task_id)
        if task is None:
            raise ReviewConflictError("not_found", "review task not found")
        now = _utcnow()
        result = await session.execute(
            update(ReviewTask)
            .where(ReviewTask.id == task.id, ReviewTask.status == ReviewTaskStatus.PENDING)
            .values(status=ReviewTaskStatus.IN_REVIEW, assigned_to=reviewer, claimed_at=now, version=ReviewTask.version + 1)
        )
        await session.commit()
        if result.rowcount == 0:
            task = await self.get_task(session, task_id)
            if task.status == ReviewTaskStatus.RESOLVED:
                raise ReviewConflictError("already_resolved", "review task already resolved")
            raise ReviewConflictError("already_claimed", f"review task claimed by {task.assigned_to}")
        task = await self.get_task(session, task_id)
        schedule_broadcast(_review_event(task, "review.claimed").to_broadcast())
        return task

    async def resolve(
        self,
        session: AsyncSession,
        task_id: str,
        reviewer: str,
        human_decision: HumanDecision,
        human_label: str | None,
        reason: str | None,
    ) -> ReviewTask:
        """5E/5F: validate, derive final quality result, persist an immutable
        decision, set the inspection final result, resolve the task."""
        task = await self.get_task(session, task_id)
        if task is None:
            raise ReviewConflictError("not_found", "review task not found")
        if task.status == ReviewTaskStatus.RESOLVED:
            raise ReviewConflictError("already_resolved", "review task already resolved")
        if task.status != ReviewTaskStatus.IN_REVIEW:
            raise ReviewConflictError("not_claimed", "review task must be claimed before resolving")
        if task.assigned_to != reviewer:
            raise ReviewConflictError("not_owner", f"task claimed by {task.assigned_to}")

        label = (human_label or "").strip() or None
        if human_decision in (HumanDecision.CONFIRM_DEFECT, HumanDecision.CORRECT_DEFECT, HumanDecision.OTHER_DEFECT) and not label:
            raise ReviewValidationError(f"{human_decision.value} requires human_label")

        final = FINAL_BY_DECISION[human_decision]
        now = _utcnow()

        result = await session.execute(
            update(ReviewTask)
            .where(
                ReviewTask.id == task.id,
                ReviewTask.status == ReviewTaskStatus.IN_REVIEW,
                ReviewTask.version == task.version,
            )
            .values(status=ReviewTaskStatus.RESOLVED, resolved_at=now, version=ReviewTask.version + 1)
        )
        if result.rowcount == 0:
            raise ReviewConflictError("already_resolved", "review task resolved concurrently")

        decision = ReviewDecision(
            review_task_id=task.id,
            inspection_id=task.inspection_id,
            reviewer=reviewer,
            ai_quality_result=task.ai_quality_result,
            ai_defects_snapshot=task.ai_defects_snapshot,
            human_decision=human_decision,
            human_label=label,
            final_quality_result=final,
            reason=reason,
        )
        session.add(decision)
        await session.execute(
            update(Inspection)
            .where(Inspection.id == task.inspection_id)
            .values(final_quality_result=final.value)
        )
        # single transaction: task resolve + decision + inspection final result
        await session.commit()
        task = await self.get_task(session, task_id)

        # Phase 7: a human decision turns the HOLD into RELEASE / REJECT.
        if task.inspection is not None:
            from .industrial_service import IndustrialService

            await IndustrialService().process_result(
                session, task.inspection,
                final_quality_result=final.value,
                process_status="completed",
                review_resolved=True,
                review_decision=human_decision.value,
                reviewed_by=reviewer,
            )
            await session.commit()

        schedule_broadcast(
            _review_event(
                task, "review.resolved", reviewer=reviewer,
                human_decision=human_decision.value, final_quality_result=final.value,
            ).to_broadcast()
        )
        return task

    async def add_correction(
        self,
        session: AsyncSession,
        task_id: str,
        reviewer: str,
        field_changed: str,
        new_value: dict,
        reason: str | None,
    ) -> ReviewCorrection:
        """5F: post-resolve revisions are appended, never written over the
        original decision."""
        task = await self.get_task(session, task_id)
        if task is None or task.decision is None:
            raise ReviewConflictError("not_found", "review decision not found for task")
        if task.status != ReviewTaskStatus.RESOLVED:
            raise ReviewConflictError("not_resolved", "corrections only apply to resolved reviews")
        old_value = getattr(task.decision, field_changed, None)
        if old_value is None:
            raise ReviewValidationError(f"unknown field '{field_changed}'")
        old_value = old_value.value if hasattr(old_value, "value") else old_value
        correction = ReviewCorrection(
            review_decision_id=task.decision.id,
            reviewer=reviewer,
            field_changed=field_changed,
            old_value={"value": old_value},
            new_value=new_value,
            reason=reason,
        )
        session.add(correction)
        await session.commit()
        return correction

    async def metrics(self, session: AsyncSession) -> dict:
        """5K (Phase 6 pre-gate): review metrics with explicit semantics.

        pending_review_count        = PENDING + IN_REVIEW (unresolved)
        average_review_wait_time_s  = mean(resolved_at - created_at) over resolved
        review_rate                 = REVIEW tasks / completed inspections
        defect_confirmation_rate    = human confirmed the AI defect /
                                      resolved reviews (renamed from
                                      ai_human_agreement_rate)
        ai_human_label_agreement_rate = among decisions WITH a human_label,
                                      share whose label equals the AI top
                                      defect class, over resolved reviews
        override_rate               = 1 - defect_confirmation_rate
        corrected_label_count       = count(CORRECT_DEFECT)
        """
        from ..metrics import metrics as rt

        task_status = await session.execute(
            select(ReviewTask.status, func.count()).group_by(ReviewTask.status)
        )
        counts = {row[0].value if hasattr(row[0], "value") else row[0]: row[1] for row in task_status}
        pending = counts.get("PENDING", 0)
        in_review = counts.get("IN_REVIEW", 0)
        resolved = counts.get("RESOLVED", 0)

        wait_row = await session.execute(
            select(func.avg(func.extract("epoch", ReviewTask.resolved_at - ReviewTask.created_at))).where(
                ReviewTask.resolved_at.is_not(None)
            )
        )
        avg_wait = wait_row.scalar_one()
        wait_seconds = round(avg_wait, 1) if avg_wait is not None else None

        decisions = (await session.execute(select(ReviewDecision))).scalars().all()
        confirm = sum(1 for d in decisions if d.human_decision == HumanDecision.CONFIRM_DEFECT)
        pass_count = sum(1 for d in decisions if d.human_decision == HumanDecision.PASS)
        corrected = sum(1 for d in decisions if d.human_decision == HumanDecision.CORRECT_DEFECT)
        label_agree = 0
        for d in decisions:
            if not d.human_label:
                continue
            top_class, _ = _top_defect(d.ai_defects_snapshot or [])
            if top_class and d.human_label == top_class:
                label_agree += 1

        rt_snap = await rt.snapshot()
        completed = rt_snap["completed_total"]

        confirmation = round(confirm / resolved, 4) if resolved else None
        total_tasks = pending + in_review + resolved
        return {
            "pending_review_count": pending + in_review,
            "pending": pending,
            "in_review": in_review,
            "resolved": resolved,
            "average_review_wait_time_s": wait_seconds,
            # review_rate = AI REVIEW inspections / completed inspections (5K)
            "review_rate": round(total_tasks / completed, 4) if completed else None,
            "defect_confirmation_rate": confirmation,
            "ai_human_label_agreement_rate": round(label_agree / resolved, 4) if resolved else None,
            "override_rate": round(1 - confirmation, 4) if confirmation is not None else None,
            "corrected_label_count": corrected,
            "pass_overrides": pass_count,
        }
