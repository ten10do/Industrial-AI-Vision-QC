"""Evaluation evidence service.

Owns the lifecycle of a stored evaluation report:

1. **ingest** -- accept a report from the signed trusted pipeline, re-verify
   its fingerprint on the server, cross-check the confusion matrix against the
   counts it claims, and append a row;
2. **retrieve** -- the latest evidence for a model version, which is what the
   quality gate judges;
3. **judge** -- run the quality gate against that evidence and write the
   verdict into the governance journal, whether it passed, held, or was not
   enforced.

The service never edits or deletes a stored evaluation. A corrected
evaluation is a new row, so an auditor can always see what a past promotion
decision was actually based on.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..evaluation.report import EVALUATION_SCHEMA_VERSION, evaluation_fingerprint
from ..mlops.quality_gate import (
    VERDICT_HOLD,
    VERDICT_NOT_ENFORCED,
    VERDICT_PASS,
    QualityGateResult,
    evaluate_quality_gate,
)
from ..models import ModelEvaluation, ModelRegistry
from ..security.auth import ROLE_ADMIN, ROLE_PIPELINE

logger = logging.getLogger(__name__)

REQUIRED_REPORT_KEYS = (
    "schema_version",
    "model",
    "dataset",
    "metrics",
    "confusion_matrix",
    "failure_cases",
    "threshold",
    "fingerprint",
)

# Journal actions written by this service.
ACTION_EVALUATION = "evaluate"
ACTION_QUALITY_GATE = "quality_gate"


class EvaluationSubmissionError(Exception):
    """A submitted report cannot be trusted as evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def verify_report(report: Any) -> tuple[dict, str]:
    """Validate an incoming report and return ``(report, recomputed_fingerprint)``.

    Every check here is a check the *server* performs. A caller that fabricates
    a report has to fabricate a self-consistent one, and the digest it quotes
    must equal the digest of the bytes it sent.
    """
    if not isinstance(report, Mapping):
        raise EvaluationSubmissionError("report_not_a_mapping", "report must be an object")
    report = dict(report)

    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise EvaluationSubmissionError(
            "report_incomplete", f"report is missing required field(s): {missing}"
        )
    if report.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationSubmissionError(
            "report_schema_mismatch",
            f"expected schema_version {EVALUATION_SCHEMA_VERSION!r}, got {report.get('schema_version')!r}",
        )

    expected = evaluation_fingerprint(report)
    claimed = str(report.get("fingerprint") or "")
    if claimed.lower() != expected:
        raise EvaluationSubmissionError(
            "report_fingerprint_mismatch",
            f"quoted fingerprint {claimed[:16]}... does not match the recomputed {expected[:16]}...",
        )

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluationSubmissionError("report_metrics_invalid", "metrics must be an object")
    matrix = report.get("confusion_matrix")
    if not (isinstance(matrix, (list, tuple)) and len(matrix) == 2
            and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in matrix)):
        raise EvaluationSubmissionError("report_confusion_matrix_invalid", "confusion_matrix must be 2x2")

    tn, fp = int(matrix[0][0]), int(matrix[0][1])
    fn, tp = int(matrix[1][0]), int(matrix[1][1])
    for name, matrix_value in (("tn", tn), ("fp", fp), ("fn", fn), ("tp", tp)):
        claimed_value = metrics.get(name)
        if claimed_value is None:
            continue
        if int(claimed_value) != matrix_value:
            raise EvaluationSubmissionError(
                "report_confusion_matrix_inconsistent",
                f"confusion_matrix says {name}={matrix_value} but metrics says {name}={int(claimed_value)}",
            )

    sample_count = metrics.get("sample_count")
    if sample_count is None:
        raise EvaluationSubmissionError("report_sample_count_missing", "metrics.sample_count is required")
    if int(sample_count) != tn + fp + fn + tp:
        raise EvaluationSubmissionError(
            "report_sample_count_inconsistent",
            f"sample_count {int(sample_count)} != tn+fp+fn+tp {tn + fp + fn + tp}",
        )

    dataset = report.get("dataset")
    if isinstance(dataset, Mapping) and dataset.get("sample_count") is not None:
        if int(dataset["sample_count"]) != int(sample_count):
            raise EvaluationSubmissionError(
                "report_dataset_count_inconsistent",
                f"dataset.sample_count {dataset['sample_count']} != metrics.sample_count {sample_count}",
            )

    model = report.get("model")
    if not isinstance(model, Mapping) or not model.get("name") or not model.get("version"):
        raise EvaluationSubmissionError("report_model_identity_missing", "model.name and model.version are required")

    return report, expected


class EvaluationService:
    # ---- ingest ----

    async def submit(
        self,
        session: AsyncSession,
        *,
        actor,
        report: Mapping,
        entry: ModelRegistry | None = None,
        model_type: str | None = None,
        report_uri: str | None = None,
        attestation_digest: str | None = None,
        request_id: str | None = None,
    ) -> ModelEvaluation:
        """Store a verified evaluation report. Append-only, never updates."""
        if not actor.has_any(ROLE_PIPELINE, ROLE_ADMIN):
            raise EvaluationSubmissionError(
                "insufficient_role",
                f"principal {actor.subject} may not submit evaluation evidence",
            )

        verified, fingerprint = verify_report(report)
        model = verified["model"]
        dataset = verified["dataset"]
        metrics = verified["metrics"]

        if entry is not None:
            if str(model["name"]) != entry.model_name or str(model["version"]) != entry.model_version:
                raise EvaluationSubmissionError(
                    "report_model_mismatch",
                    f"report is for {model['name']}@{model['version']} but the registry entry is "
                    f"{entry.model_name}@{entry.model_version}",
                )

        evaluation_time = None
        raw_time = verified.get("evaluation_time")
        if isinstance(raw_time, str):
            try:
                evaluation_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            except ValueError:
                raise EvaluationSubmissionError(
                    "report_evaluation_time_invalid", f"evaluation_time {raw_time!r} is not ISO 8601"
                ) from None
        if evaluation_time is None:
            raise EvaluationSubmissionError(
                "report_evaluation_time_missing",
                "evaluation_time is required so that evidence age can be judged",
            )

        confusion = verified["confusion_matrix"]
        latency = metrics.get("latency") if isinstance(metrics.get("latency"), Mapping) else {}

        row = ModelEvaluation(
            registry_id=entry.id if entry is not None else None,
            model_name=str(model["name"]),
            model_version=str(model["version"]),
            model_type=model_type if model_type is not None else (entry.model_type if entry else None),
            task=str(verified.get("task") or "anomaly_detection"),
            dataset_name=str(dataset.get("name")),
            dataset_split=str(dataset.get("split")),
            sample_count=int(metrics["sample_count"]),
            threshold=float(verified["threshold"]) if isinstance(verified.get("threshold"), (int, float)) else None,
            precision=_opt_float(metrics.get("precision")),
            recall=_opt_float(metrics.get("recall")),
            f1=_opt_float(metrics.get("f1")),
            false_accept_rate=_opt_float(metrics.get("false_accept_rate")),
            false_reject_rate=_opt_float(metrics.get("false_reject_rate")),
            latency_p95_ms=_opt_float(latency.get("p95_ms")),
            true_positive=int(confusion[1][1]),
            true_negative=int(confusion[0][0]),
            false_positive=int(confusion[0][1]),
            false_negative=int(confusion[1][0]),
            report=verified,
            report_sha256=fingerprint,
            report_uri=report_uri,
            attested_by=actor.subject,
            attestation_digest=attestation_digest,
            evaluation_time=evaluation_time,
        )
        session.add(row)
        await session.flush()

        await self._audit(
            session, entry=entry, actor=actor, action=ACTION_EVALUATION, outcome="APPLIED",
            payload={
                "evaluation_id": str(row.id),
                "model_name": row.model_name,
                "model_version": row.model_version,
                "report_sha256": fingerprint,
                "report_uri": report_uri,
                "dataset": {"name": row.dataset_name, "split": row.dataset_split},
                "sample_count": row.sample_count,
                "threshold": row.threshold,
                "metrics": {
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                    "false_accept_rate": row.false_accept_rate,
                    "false_reject_rate": row.false_reject_rate,
                    "latency_p95_ms": row.latency_p95_ms,
                },
                "confusion_matrix": confusion,
            },
            request_id=request_id,
        )
        return row

    # ---- reads ----

    async def get(self, session: AsyncSession, evaluation_id: uuid.UUID) -> ModelEvaluation | None:
        return await session.get(ModelEvaluation, evaluation_id)

    async def latest(
        self, session: AsyncSession, *, model_name: str, model_version: str
    ) -> ModelEvaluation | None:
        """The most recent evaluation for a version, by evaluation time.

        Ordered by the evaluation's own timestamp rather than by insertion
        order, so a backfilled older report cannot masquerade as the newest
        evidence.
        """
        stmt = (
            select(ModelEvaluation)
            .where(
                ModelEvaluation.model_name == model_name,
                ModelEvaluation.model_version == model_version,
            )
            .order_by(ModelEvaluation.evaluation_time.desc(), ModelEvaluation.created_at.desc())
            .limit(1)
        )
        return (await session.execute(stmt)).scalars().first()

    async def list(
        self,
        session: AsyncSession,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        registry_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[ModelEvaluation]:
        stmt = select(ModelEvaluation).order_by(ModelEvaluation.created_at.desc()).limit(max(1, min(limit, 200)))
        if model_name:
            stmt = stmt.where(ModelEvaluation.model_name == model_name)
        if model_version:
            stmt = stmt.where(ModelEvaluation.model_version == model_version)
        if registry_id is not None:
            stmt = stmt.where(ModelEvaluation.registry_id == registry_id)
        return list((await session.execute(stmt)).scalars().all())

    # ---- gate ----

    async def quality_gate_for(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        actor=None,
        request_id: str | None = None,
        record_audit: bool = False,
    ) -> QualityGateResult:
        """Judge a registry entry against the stored evaluation evidence.

        Read-only by default. The gate evaluation that lands in the governance
        journal is the one attached to the promotion decision it affected: the
        ``promote`` row carries the full verdict, the failed rules, the policy
        identity and the evidence fingerprint in its payload. Writing a second
        row for the same attempt would leave an auditor joining two records to
        reconstruct one decision.

        Set ``record_audit=True`` to append a standalone ``quality_gate`` row
        instead; the caller then owns the commit.
        """
        evidence_row = await self.latest(session, model_name=entry.model_name,
                                         model_version=entry.model_version)
        evidence = evidence_row.report if evidence_row is not None else None

        age_days = None
        if evidence_row is not None:
            stamp = _as_utc(evidence_row.evaluation_time)
            if stamp is not None:
                age_days = (_utc_now() - stamp).total_seconds() / 86400.0

        result = evaluate_quality_gate(
            model_type=entry.model_type,
            model_name=entry.model_name,
            model_version=entry.model_version,
            evidence=evidence,
            evidence_age_days=age_days,
        )
        if evidence_row is not None and result.evidence is None:
            result.evidence = {
                "evaluation_id": str(evidence_row.id),
                "schema_version": evidence_row.report.get("schema_version"),
                "fingerprint": evidence_row.report_sha256,
                "model_name": evidence_row.model_name,
                "model_version": evidence_row.model_version,
                "dataset": evidence_row.dataset_name,
                "split": evidence_row.dataset_split,
                "sample_count": evidence_row.sample_count,
                "threshold": evidence_row.threshold,
                "evaluation_time": evidence_row.evaluation_time.isoformat() if evidence_row.evaluation_time else None,
            }
        if evidence_row is not None:
            result.evidence = {**(result.evidence or {}), "evaluation_id": str(evidence_row.id)}

        if record_audit:
            await self._audit(
                session, entry=entry, actor=actor, action=ACTION_QUALITY_GATE,
                outcome=_gate_outcome(result.verdict),
                payload={"quality_gate": result.to_dict()},
                request_id=request_id,
            )
        return result

    # ---- internals ----

    async def _audit(
        self,
        session: AsyncSession,
        *,
        entry: ModelRegistry | None,
        actor,
        action: str,
        outcome: str,
        payload: dict,
        request_id: str | None,
    ) -> None:
        from ..services.registry_service import get_registry_service

        await get_registry_service().audit(
            session, action=action, outcome=outcome, entry=entry, actor=actor,
            payload=payload, request_id=request_id,
        )


def _gate_outcome(verdict: str) -> str:
    if verdict == VERDICT_PASS:
        return "PASSED"
    if verdict == VERDICT_NOT_ENFORCED:
        return "NOT_ENFORCED"
    if verdict == VERDICT_HOLD:
        return "HELD"
    return "ERROR"


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def get_evaluation_service() -> EvaluationService:
    return EvaluationService()
