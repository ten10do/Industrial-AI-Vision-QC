"""MLOps monitoring API (8G / 8H / 8I).

GET /api/v1/model-metrics?model_version=&time_from=&time_to=
    production monitoring aggregate: inference_count / error_count /
    latency avg+p95 / confidence_distribution / defect_distribution /
    anomaly_score_distribution / review_rate / human override / corrected

GET /api/v1/human-feedback?model_version=&defect_type=&line=&station=&time_from=&time_to=
    Human Review ground-truth metrics sliced by model_version / defect /
    line / station / window: defect_confirmation_rate /
    ai_human_label_agreement_rate / pass_override_rate / corrected_label_rate

GET /api/v1/drift?model_version=&baseline_days=&window=100
    baseline vs current window on confidence / defect / anomaly / review
    rate -> NORMAL / WARNING / CRITICAL (data drift, NOT quality
    degradation; the distinction is documented in 8I)
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..mlops.drift import (
    classify_ks,
    classify_psi,
    defect_distribution_delta,
    ks_statistic,
    psi,
    review_rate_delta,
)
from ..models import Inspection, ReviewDecision, ReviewTask
from ..security.auth import require_any_authenticated

router = APIRouter(prefix="/api/v1", tags=["mlops"])


async def _inspections_in(session: AsyncSession, *, model_version: str | None, time_from: datetime | None,
                          time_to: datetime | None, limit: int = 2000) -> list[Inspection]:
    from sqlalchemy.orm import selectinload

    stmt = (
        select(Inspection)
        .options(selectinload(Inspection.defects))
        .order_by(Inspection.created_at.desc())
        .limit(limit)
    )
    if model_version:
        stmt = stmt.where(Inspection.model_version == model_version)
    if time_from:
        stmt = stmt.where(Inspection.created_at >= time_from)
    if time_to:
        stmt = stmt.where(Inspection.created_at <= time_to)
    return list((await session.execute(stmt)).scalars().all())


def _bins(values: list[float], n: int = 10, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    out = [0] * n
    span = (hi - lo) or 1.0
    for v in values:
        idx = int((v - lo) / span * n)
        idx = max(0, min(n - 1, idx))
        out[idx] += 1
    return out


@router.get("/model-metrics", dependencies=[Depends(require_any_authenticated())])
async def model_metrics(
    model_version: str | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    rows = await _inspections_in(session, model_version=model_version, time_from=time_from, time_to=time_to)
    completed = [i for i in rows if i.status == "completed"]
    failed = [i for i in rows if i.status == "failed"]
    review = [i for i in completed if i.quality_result == "REVIEW"]
    confs: list[float] = []
    defects: dict[str, int] = {}
    anom: list[float] = []
    lat: list[float] = []
    for i in completed:
        if i.inference_latency_ms is not None:
            lat.append(i.inference_latency_ms)
        if i.anomaly_score is not None:
            anom.append(i.anomaly_score)
        for d in i.defects:
            confs.append(float(d.confidence))
            defects[d.class_name] = defects.get(d.class_name, 0) + 1

    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else None
    return {
        "model_version": model_version,
        "window_count": len(completed),
        "inference_count": len(completed),
        "error_count": len(failed),
        "error_rate": round(len(failed) / max(1, len(completed) + len(failed)), 4),
        "inference_latency_avg_ms": round(statistics.fmean(lat), 2) if lat else None,
        "inference_latency_p95_ms": round(p95, 2) if p95 else None,
        "review_rate": round(len(review) / max(1, len(completed)), 4),
        "confidence_distribution": _bins(confs),
        "defect_distribution": defects,
        "anomaly_score_distribution": _bins(anom),
    }


@router.get("/human-feedback", dependencies=[Depends(require_any_authenticated())])
async def human_feedback(
    model_version: str | None = None,
    defect_type: str | None = None,
    line: str | None = None,
    station: str | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    stmt = (
        select(ReviewDecision, ReviewTask, Inspection)
        .join(ReviewTask, ReviewTask.id == ReviewDecision.review_task_id)
        .join(Inspection, Inspection.id == ReviewDecision.inspection_id)
        .order_by(ReviewDecision.created_at.desc())
        .limit(5000)
    )
    if model_version:
        stmt = stmt.where(Inspection.model_version == model_version)
    if time_from:
        stmt = stmt.where(ReviewDecision.created_at >= time_from)
    if time_to:
        stmt = stmt.where(ReviewDecision.created_at <= time_to)
    rows = (await session.execute(stmt)).all()

    resolved = 0
    confirm = 0
    label_agree = 0
    pass_override = 0
    corrected = 0
    other = 0
    per_defect: dict[str, dict] = {}

    for decision, task, insp in rows:
        # slice by line/station from the task denormalized columns
        if line and task.production_line != line:
            continue
        if station and task.station != station:
            continue
        hd = decision.human_decision.value if hasattr(decision.human_decision, "value") else str(decision.human_decision)
        label = decision.human_label
        if defect_type and label != defect_type:
            continue
        resolved += 1
        if hd == "CONFIRM_DEFECT":
            confirm += 1
            # ai top defect label agreement (Phase 7 metric)
            snapshot = decision.ai_defects_snapshot or []
            top = snapshot[0].get("class_name") if snapshot else None
            if label and top and label == top:
                label_agree += 1
        elif hd == "PASS":
            pass_override += 1
        elif hd == "CORRECT_DEFECT":
            corrected += 1
        else:
            other += 1

        # per-defect slice (by human label or ai top class)
        key = label or (task.ai_defects_snapshot[0].get("class_name") if task.ai_defects_snapshot else "unknown")
        d = per_defect.setdefault(key, {"resolved": 0, "confirm": 0, "agree": 0, "pass": 0, "corrected": 0})
        d["resolved"] += 1
        if hd == "CONFIRM_DEFECT":
            d["confirm"] += 1
            snapshot = decision.ai_defects_snapshot or []
            top = snapshot[0].get("class_name") if snapshot else None
            if label and top and label == top:
                d["agree"] += 1
        elif hd == "PASS":
            d["pass"] += 1
        elif hd == "CORRECT_DEFECT":
            d["corrected"] += 1

    def _rate(n: int) -> float | None:
        return round(n / resolved, 4) if resolved else None

    per_defect_out = {
        k: {
            "defect_confirmation_rate": _rate(v["confirm"]),
            "ai_human_label_agreement_rate": _rate(v["agree"]),
            "pass_override_rate": _rate(v["pass"]),
            "corrected_label_rate": _rate(v["corrected"]),
            "resolved": v["resolved"],
        }
        for k, v in sorted(per_defect.items())
    }
    return {
        "filters": {"model_version": model_version, "defect_type": defect_type, "line": line, "station": station},
        "resolved": resolved,
        "defect_confirmation_rate": _rate(confirm),
        "ai_human_label_agreement_rate": _rate(label_agree),
        "pass_override_rate": _rate(pass_override),
        "corrected_label_rate": _rate(corrected),
        "per_defect": per_defect_out,
    }


@router.get("/drift", dependencies=[Depends(require_any_authenticated())])
async def drift(
    model_version: str | None = None,
    baseline_days: int = Query(default=7, ge=1),
    window: int = Query(default=200, ge=10),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Compare a baseline window (older) with the current window on:
    confidence PSI, anomaly score PSI, defect distribution delta, review
    rate delta. Output NORMAL / WARNING / CRITICAL per signal (8I)."""
    now = datetime.now(timezone.utc)
    baseline_start = now - timedelta(days=baseline_days * 2)

    async def _fetch(from_dt: datetime, to_dt: datetime) -> list[Inspection]:
        from sqlalchemy.orm import selectinload

        stmt = (
            select(Inspection)
            .options(selectinload(Inspection.defects))
            .where(Inspection.created_at >= from_dt, Inspection.created_at <= to_dt)
            .order_by(Inspection.created_at.asc())
            .limit(window)
        )
        if model_version:
            stmt = stmt.where(Inspection.model_version == model_version)
        return list((await session.execute(stmt)).scalars().all())

    baseline = await _fetch(baseline_start, now - timedelta(days=baseline_days))
    current = await _fetch(now - timedelta(days=baseline_days), now)

    def _conf(i: Inspection) -> float:
        return float(i.defects[0].confidence) if i.defects else 0.0

    b_conf = [_conf(i) for i in baseline if i.defects]
    c_conf = [_conf(i) for i in current if i.defects]
    b_anom = [float(i.anomaly_score) for i in baseline if i.anomaly_score is not None]
    c_anom = [float(i.anomaly_score) for i in current if i.anomaly_score is not None]

    b_defect: dict[str, float] = {}
    c_defect: dict[str, float] = {}
    for i in baseline:
        for d in i.defects:
            b_defect[d.class_name] = b_defect.get(d.class_name, 0) + 1
    for i in current:
        for d in i.defects:
            c_defect[d.class_name] = c_defect.get(d.class_name, 0) + 1
    total_b = sum(b_defect.values()) or 1
    total_c = sum(c_defect.values()) or 1
    b_defect = {k: v / total_b for k, v in b_defect.items()}
    c_defect = {k: v / total_c for k, v in c_defect.items()}

    def _rate_of(items: list[Inspection]) -> float:
        comp = [i for i in items if i.status == "completed"]
        return len([i for i in comp if i.quality_result == "REVIEW"]) / max(1, len(comp))

    signals = {
        "confidence_psi": {"score": round(psi(b_conf, c_conf), 4), "level": classify_psi(psi(b_conf, c_conf)) if b_conf and c_conf else "NORMAL", "baseline_n": len(b_conf), "current_n": len(c_conf)},
        "anomaly_score_psi": {"score": round(psi(b_anom, c_anom, lo=0.0, hi=1.0), 4), "level": classify_psi(psi(b_anom, c_anom, lo=0.0, hi=1.0)) if b_anom and c_anom else "NORMAL", "baseline_n": len(b_anom), "current_n": len(c_anom)},
        "defect_distribution": {"max_delta": round(max((abs(c_defect.get(k, 0.0) - b_defect.get(k, 0.0)) for k in set(b_defect) | set(c_defect)), default=0.0), 4), "level": defect_distribution_delta(b_defect, c_defect)},
        "review_rate": {"baseline": round(_rate_of(baseline), 4), "current": round(_rate_of(current), 4), "level": review_rate_delta(_rate_of(baseline), _rate_of(current))},
    }
    levels = [s["level"] for s in signals.values()]
    overall = "CRITICAL" if "CRITICAL" in levels else ("WARNING" if "WARNING" in levels else "NORMAL")
    return {
        "model_version": model_version,
        "baseline_window": {"from": baseline_start.isoformat(), "to": (now - timedelta(days=baseline_days)).isoformat(), "n": len(baseline)},
        "current_window": {"from": (now - timedelta(days=baseline_days)).isoformat(), "to": now.isoformat(), "n": len(current)},
        "overall": overall,
        "signals": signals,
        # 8I semantic boundary: drift is DATA drift, not quality degradation.
        "note": "data drift only; quality degradation requires human-review ground truth",
    }
