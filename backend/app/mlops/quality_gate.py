"""Model Quality Gate (evaluation evidence).

A candidate model does not become production-ready because training finished.
It becomes a candidate for *human approval* only after an offline evaluation
has been produced, stored, and judged against a written policy.

This module owns the judgement. It is:

* **deterministic** -- the same evidence and the same policy always produce
  the same verdict; no model, no sampling, no random tie-break;
* **configurable** -- every number lives in ``quality_gate_policy.yaml``, a
  file the process reads and pins by SHA256, never in a request body;
* **testable** -- the evaluator is a pure function over (policy, evidence);
* **auditable** -- the verdict carries the policy identity, the policy digest,
  the evaluated model version, the evidence fingerprint, every failed rule in
  machine and human form, and a timestamp;
* **fail-closed** -- a missing policy, a missing evidence record, a
  structurally incomplete report, or an unreadable metric all produce HOLD.
  There is no "degrade to a warning" path.

Relationship to the promotion gate
----------------------------------
``app.mlops.promotion_gate`` answers "are the attested metrics good enough?".
This module answers the harder question "is there an evaluation that proves
it, and does that evaluation survive a quality policy?". Both must pass. They
are separate because they consume different evidence: the promotion gate
reads attested aggregate metrics, this gate reads a prediction-level
evaluation report with a confusion matrix and failure cases behind it.

There is deliberately no override path. The scope of enforcement is declared
in the policy file (``enforcement.enforced_model_types``), which is
hash-pinned and reviewed like code. A gate that can be talked out of its own
verdict at runtime is not a gate.

No LLM decides whether a model ships. The verdict is arithmetic over a
written policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import get_settings, resolve_path
from .gate_policy import MetricRule, ModelTypePolicy, PolicyError, resolve_rules, sha256_of_file

SCHEMA_VERSION = "ivqc_quality_gate_policy_v1"

VERDICT_PASS = "PASS"
VERDICT_HOLD = "HOLD"
VERDICT_NOT_ENFORCED = "NOT_ENFORCED"

# Metrics this gate knows how to read out of an evaluation report. Anything a
# policy names outside this set is a policy error, not a silently ignored rule.
EVIDENCE_METRICS = (
    "sample_count",
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "false_accept_rate",
    "false_reject_rate",
    "error_rate",
    "threshold",
    "latency_p95_ms",
)

# Structural requirements, keyed by the name a policy uses to require them.
STRUCTURAL_REQUIREMENTS = {
    "require_confusion_matrix": "confusion matrix",
    "require_failure_cases": "false-positive / false-negative failure cases",
    "require_threshold_recorded": "operating threshold",
    "require_sample_count": "sample count",
    "require_score_distributions": "per-class score distributions",
}


class QualityGatePolicyError(PolicyError):
    """The quality gate policy cannot be trusted. Callers must fail closed."""


# ---------------------------------------------------------------- policy ----


@dataclass(frozen=True)
class Enforcement:
    """Where the gate is actually applied.

    Scope is policy, not runtime configuration. An empty
    ``enforced_model_types`` is a truthful statement that this repository has
    not yet calibrated a quality gate for any model family, and it is
    reported on every verdict rather than hidden.
    """

    enforced_model_types: tuple[str, ...]
    require_evaluation_evidence: bool

    def applies_to(self, model_type: str) -> bool:
        return model_type in self.enforced_model_types

    def to_dict(self) -> dict:
        return {
            "enforced_model_types": list(self.enforced_model_types),
            "require_evaluation_evidence": self.require_evaluation_evidence,
            "enabled": bool(self.enforced_model_types),
        }


@dataclass(frozen=True)
class QualityGatePolicy:
    policy_id: str
    sha256: str
    pinned: bool
    status: str
    notes: tuple[str, ...]
    enforcement: Enforcement
    requirements: dict[str, bool]
    max_evidence_age_days: float | None
    model_types: dict[str, ModelTypePolicy]

    def rules_for(self, model_type: str) -> tuple[MetricRule, ...]:
        return self.model_types[model_type].rules

    def to_dict(self, model_type: str | None = None) -> dict:
        out = {
            "policy_id": self.policy_id,
            "policy_sha256": self.sha256,
            "policy_pinned": self.pinned,
            "policy_status": self.status,
            "policy_path": str(quality_gate_policy_path()),
            "policy_notes": list(self.notes),
            "enforcement": self.enforcement.to_dict(),
            "evidence_requirements": dict(self.requirements),
            "max_evidence_age_days": self.max_evidence_age_days,
            "model_types": sorted(self.model_types),
        }
        if model_type is not None and model_type in self.model_types:
            out["thresholds_used"] = self.model_types[model_type].thresholds()
            out["rule_sources"] = {rule.name: rule.source for rule in self.rules_for(model_type)}
        return out


def quality_gate_policy_path() -> Path:
    return resolve_path(get_settings().quality_gate_policy_path)


def _parse_rule(name: str, raw: Any, *, status: str, model_type: str) -> MetricRule:
    if name not in EVIDENCE_METRICS:
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} is not a readable evaluation metric; "
            f"known: {list(EVIDENCE_METRICS)}"
        )
    if not isinstance(raw, dict):
        raise QualityGatePolicyError(f"quality gate policy: {model_type}.{name} must be a mapping")
    direction = raw.get("direction")
    if direction not in ("min", "max"):
        raise QualityGatePolicyError(f"quality gate policy: {model_type}.{name}.direction must be min or max")
    try:
        value = float(raw["value"])
        bound = float(raw["bound"])
    except KeyError as exc:
        raise QualityGatePolicyError(f"quality gate policy: {model_type}.{name} missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} value/bound must be numeric"
        ) from exc
    if direction == "min" and value < bound:
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} value {value} below hard floor {bound}"
        )
    if direction == "max" and value > bound:
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} value {value} above hard ceiling {bound}"
        )
    # A rule with no provenance is a rule nobody can defend. Every entry must
    # say where its number came from.
    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} must carry a 'source' explaining the number"
        )
    if status == "production" and "demo" in source.lower():
        raise QualityGatePolicyError(
            f"quality gate policy: {model_type}.{name} still cites a demo/development value "
            f"({source!r}) while the policy declares status=production"
        )
    rule = MetricRule(name=name, direction=direction, value=value, bound=bound, source=source.strip())
    return rule


def load_quality_gate_policy(path: Path | None = None) -> QualityGatePolicy:
    import yaml

    p = path or quality_gate_policy_path()
    if not p.exists():
        raise QualityGatePolicyError(f"quality gate policy not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is fatal
        raise QualityGatePolicyError(f"quality gate policy unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise QualityGatePolicyError("quality gate policy must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise QualityGatePolicyError(f"quality gate policy schema mismatch: {data.get('schema_version')!r}")

    digest = sha256_of_file(p)
    expected = (get_settings().quality_gate_policy_sha256 or "").strip().lower()
    if expected and expected != digest:
        raise QualityGatePolicyError(
            f"quality gate policy hash pin mismatch: expected {expected}, got {digest}. "
            "Refusing to evaluate any promotion against an unverified policy."
        )

    status = str(data.get("status") or "development").strip().lower()
    if status not in ("development", "production"):
        raise QualityGatePolicyError(
            f"quality gate policy: status must be development or production, got {status!r}"
        )

    raw_enforcement = data.get("enforcement") or {}
    if not isinstance(raw_enforcement, dict):
        raise QualityGatePolicyError("quality gate policy: enforcement must be a mapping")
    raw_types = raw_enforcement.get("enforced_model_types")
    if raw_types is None:
        raise QualityGatePolicyError(
            "quality gate policy: enforcement.enforced_model_types must be declared "
            "(use [] to declare that no model family is enforced yet)"
        )
    if not isinstance(raw_types, list) or any(not isinstance(item, str) for item in raw_types):
        raise QualityGatePolicyError(
            "quality gate policy: enforcement.enforced_model_types must be a list of model type names"
        )
    enforcement = Enforcement(
        enforced_model_types=tuple(sorted({item.strip() for item in raw_types if item.strip()})),
        require_evaluation_evidence=bool(raw_enforcement.get("require_evaluation_evidence", True)),
    )

    raw_requirements = data.get("evidence_requirements") or {}
    if not isinstance(raw_requirements, dict):
        raise QualityGatePolicyError("quality gate policy: evidence_requirements must be a mapping")
    unknown_requirements = sorted(set(raw_requirements) - set(STRUCTURAL_REQUIREMENTS))
    if unknown_requirements:
        raise QualityGatePolicyError(
            f"quality gate policy: unknown evidence requirement(s) {unknown_requirements}; "
            f"known: {sorted(STRUCTURAL_REQUIREMENTS)}"
        )
    requirements = {str(k): bool(v) for k, v in raw_requirements.items()}

    raw_freshness = data.get("freshness") or {}
    if not isinstance(raw_freshness, dict):
        raise QualityGatePolicyError("quality gate policy: freshness must be a mapping")
    raw_age = raw_freshness.get("max_evidence_age_days")
    max_evidence_age_days: float | None
    if raw_age is None:
        max_evidence_age_days = None
    else:
        try:
            max_evidence_age_days = float(raw_age)
        except (TypeError, ValueError) as exc:
            raise QualityGatePolicyError(
                "quality gate policy: freshness.max_evidence_age_days must be numeric or null"
            ) from exc
        if max_evidence_age_days <= 0:
            raise QualityGatePolicyError(
                "quality gate policy: freshness.max_evidence_age_days must be > 0"
            )

    raw_metrics = data.get("metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise QualityGatePolicyError("quality gate policy has no metrics section")
    model_types: dict[str, ModelTypePolicy] = {}
    for model_type, metrics in raw_metrics.items():
        if not isinstance(metrics, dict) or not metrics:
            raise QualityGatePolicyError(f"quality gate policy: {model_type} has no metrics")
        model_types[str(model_type)] = ModelTypePolicy(
            rules=tuple(_parse_rule(str(name), raw, status=status, model_type=str(model_type))
                        for name, raw in metrics.items())
        )

    notes = data.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]

    return QualityGatePolicy(
        policy_id=str(data.get("policy_id") or "unidentified-quality-gate-policy"),
        sha256=digest,
        pinned=bool(expected),
        status=status,
        notes=tuple(str(note) for note in notes),
        enforcement=enforcement,
        requirements=requirements,
        max_evidence_age_days=max_evidence_age_days,
        model_types=model_types,
    )


_POLICY: QualityGatePolicy | None = None


def get_quality_gate_policy(path: Path | None = None) -> QualityGatePolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = load_quality_gate_policy(path)
    return _POLICY


def reset_quality_gate_policy_cache() -> None:
    global _POLICY
    _POLICY = None


def safe_get_quality_gate_policy(path: Path | None = None) -> tuple[QualityGatePolicy | None, str | None]:
    """Never raises. A policy failure becomes a HOLD, never a 500."""
    try:
        return get_quality_gate_policy(path), None
    except PolicyError as exc:
        return None, str(exc)


# ---------------------------------------------------------------- result ----


@dataclass
class QualityGateResult:
    verdict: str = VERDICT_HOLD
    passed: bool = False
    enforced: bool = True
    checks: list[dict] = field(default_factory=list)
    failed_rules: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    policy: dict | None = None
    evidence: dict | None = None
    reason: str | None = None
    timestamp: str | None = None

    @property
    def held(self) -> bool:
        return self.verdict == VERDICT_HOLD

    @property
    def allows_promotion(self) -> bool:
        """True for PASS and for an explicitly unenforced scope.

        NOT_ENFORCED is not a pass: ``passed`` stays false so that nobody can
        read this gate as having certified an unevaluated model. What it does
        mean is that this gate raises no objection, and the caller must record
        the exemption.
        """
        return self.verdict in (VERDICT_PASS, VERDICT_NOT_ENFORCED)

    def failed_rule_codes(self) -> list[str]:
        return [rule["rule"] for rule in self.failed_rules]

    def warning_codes(self) -> list[str]:
        return [warning["rule"] for warning in self.warnings]

    def to_dict(self) -> dict:
        out = {
            "verdict": self.verdict,
            "passed": self.passed,
            "enforced": self.enforced,
            "allows_promotion": self.allows_promotion,
            "checks": self.checks,
            "failed_rules": self.failed_rules,
            "failed_rule_codes": self.failed_rule_codes(),
            "warnings": self.warnings,
            "warning_codes": self.warning_codes(),
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }
        if self.policy is not None:
            out["policy"] = self.policy
        if self.evidence is not None:
            out["evidence"] = self.evidence
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def _check(name: str, ok: bool, got: Any, required: Any, category: str = "metric") -> dict:
    return {"check": name, "passed": bool(ok), "got": got, "required": required, "category": category}


def _fail(result: QualityGateResult, rule: str, message: str, **extra: Any) -> None:
    entry = {"rule": rule, "message": message}
    entry.update(extra)
    result.failed_rules.append(entry)
    result.checks.append(_check(rule, False, extra.get("got"), extra.get("required"), category=extra.get("category", "metric")))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ------------------------------------------------------------- evidence ----


def evidence_metrics(report: Mapping | None) -> dict:
    """Flatten an evaluation report into the metric names a policy can name.

    Absent values stay absent. Nothing is defaulted to zero, because a zero
    that means "not measured" is indistinguishable from a zero that means
    "measured and terrible".
    """
    if not isinstance(report, Mapping):
        return {}
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    latency = metrics.get("latency") if isinstance(metrics.get("latency"), Mapping) else {}
    out: dict[str, Any] = {}
    for name in EVIDENCE_METRICS:
        if name == "latency_p95_ms":
            value = latency.get("p95_ms")
        elif name == "threshold":
            value = report.get("threshold")
        else:
            value = metrics.get(name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def evidence_problems(report: Mapping | None, requirements: Mapping[str, bool]) -> list[str]:
    """Structural gaps that make a report unusable as gate evidence."""
    if not isinstance(report, Mapping):
        return ["evidence_missing"]

    problems: list[str] = []
    matrix = report.get("confusion_matrix")
    matrix_ok = (
        isinstance(matrix, (list, tuple))
        and len(matrix) == 2
        and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in matrix)
    )
    if requirements.get("require_confusion_matrix", True) and not matrix_ok:
        problems.append("evidence_incomplete:confusion_matrix")

    cases = report.get("failure_cases")
    cases_ok = isinstance(cases, Mapping) and isinstance(cases.get("false_positive"), list) \
        and isinstance(cases.get("false_negative"), list)
    if requirements.get("require_failure_cases", True) and not cases_ok:
        problems.append("evidence_incomplete:failure_cases")

    threshold_ok = isinstance(report.get("threshold"), (int, float))
    if requirements.get("require_threshold_recorded", True) and not threshold_ok:
        problems.append("evidence_incomplete:threshold")

    sample_count_ok = isinstance((report.get("metrics") or {}).get("sample_count"), (int, float)) \
        if isinstance(report.get("metrics"), Mapping) else False
    if requirements.get("require_sample_count", True) and not sample_count_ok:
        problems.append("evidence_incomplete:sample_count")

    if requirements.get("require_score_distributions", False):
        distributions = report.get("score_distributions")
        if not isinstance(distributions, Mapping) or not distributions.get("positive_class"):
            problems.append("evidence_incomplete:score_distributions")

    model = report.get("model")
    if not isinstance(model, Mapping) or not model.get("name") or not model.get("version"):
        problems.append("evidence_incomplete:model_identity")

    return problems


# -------------------------------------------------------------- verdict ----


def evaluate_quality_gate(
    *,
    model_type: str,
    model_name: str | None = None,
    model_version: str | None = None,
    evidence: Mapping | None = None,
    policy: QualityGatePolicy | None = None,
    evidence_age_days: float | None = None,
    timestamp: str | None = None,
) -> QualityGateResult:
    """Judge a candidate against the quality gate policy.

    Returns PASS, HOLD, or NOT_ENFORCED. ``passed`` is true only for PASS.
    NOT_ENFORCED means the policy does not yet cover this model family: the
    promotion may proceed, and both the verdict and the policy identity are
    written to the audit journal so that the exemption is visible.
    """
    result = QualityGateResult(timestamp=timestamp or _utc_now())

    if policy is None:
        policy, error = safe_get_quality_gate_policy()
        if policy is None:
            result.enforced = True
            result.verdict = VERDICT_HOLD
            result.passed = False
            result.reason = "quality gate policy unavailable; failing closed"
            _fail(result, "policy_unavailable", f"quality gate policy could not be loaded: {error}",
                  category="policy")
            return result

    if not policy.pinned:
        # Consistent with the promotion gate: an unpinned policy is surfaced on
        # every verdict and is never a passing configuration for production,
        # but it does not by itself hold a promotion. Holding on the pin alone
        # would make every development promotion impossible, and a control that
        # nobody can satisfy is a control that gets switched off.
        result.checks.append(_check("policy_pinned", False, False, True, category="policy"))
        result.warnings.append({
            "rule": "policy_pin_missing",
            "message": (
                "the quality gate policy is not pinned by SHA256 "
                "(IVQC_QUALITY_GATE_POLICY_SHA256 is empty); the file can be edited after the "
                "fact and this is not a passing configuration for production"
            ),
            "got": False,
            "required": True,
            "category": "policy",
        })

    model_type = str(model_type or "")
    if model_type not in policy.model_types and policy.enforcement.applies_to(model_type):
        result.enforced = True
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = f"no quality gate rules declared for model_type {model_type!r}"
        _fail(result, "unknown_model_type",
              f"model_type {model_type!r} is enforced but has no rules in the policy",
              got=model_type, required=sorted(policy.model_types), category="policy")
        result.policy = policy.to_dict()
        return result

    result.policy = policy.to_dict(model_type if model_type in policy.model_types else None)

    if not policy.enforcement.applies_to(model_type):
        result.enforced = False
        result.verdict = VERDICT_NOT_ENFORCED
        result.passed = False
        result.reason = (
            f"the quality gate policy does not enforce model_type {model_type!r}; "
            "promotion is not gated on offline evaluation evidence for this model family"
        )
        result.checks.append(
            _check("enforcement_scope", True, model_type,
                   list(policy.enforcement.enforced_model_types), category="policy")
        )
        return result

    result.enforced = True

    if evidence is None and policy.enforcement.require_evaluation_evidence:
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = "no stored evaluation evidence for this model version"
        _fail(
            result,
            "evaluation_evidence_missing",
            "no offline evaluation report is stored for this model version; "
            "a candidate cannot be judged, so it cannot pass",
            got=None,
            required="an evaluation report",
            category="evidence",
        )
        return result

    problems = evidence_problems(evidence, policy.requirements)
    if problems:
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = "the stored evaluation evidence is structurally incomplete"
        for problem in problems:
            _fail(result, problem, f"evaluation evidence is unusable: {problem}", category="evidence")
        result.evidence = _evidence_ref(evidence)
        return result

    evidence_model = (evidence or {}).get("model") or {}
    if model_name and evidence_model.get("name") and str(evidence_model["name"]) != str(model_name):
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = "the stored evaluation belongs to a different model"
        _fail(result, "evidence_model_mismatch",
              f"evidence is for {evidence_model.get('name')!r}, gate asked about {model_name!r}",
              got=evidence_model.get("name"), required=model_name, category="evidence")
        result.evidence = _evidence_ref(evidence)
        return result
    if model_version and evidence_model.get("version") and str(evidence_model["version"]) != str(model_version):
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = "the stored evaluation belongs to a different model version"
        _fail(result, "evidence_version_mismatch",
              f"evidence is for version {evidence_model.get('version')!r}, "
              f"gate asked about {model_version!r}",
              got=evidence_model.get("version"), required=model_version, category="evidence")
        result.evidence = _evidence_ref(evidence)
        return result

    max_age = policy.max_evidence_age_days
    if max_age is not None and evidence_age_days is not None and evidence_age_days > max_age:
        result.verdict = VERDICT_HOLD
        result.passed = False
        result.reason = "the stored evaluation evidence is older than the policy allows"
        _fail(result, "evidence_stale",
              f"evidence is {evidence_age_days:.1f} days old, policy allows {max_age:g}",
              got=evidence_age_days, required=max_age, category="evidence")
        result.evidence = _evidence_ref(evidence)
        return result

    metrics = evidence_metrics(evidence)
    result.metrics = metrics
    result.evidence = _evidence_ref(evidence)

    rules = policy.rules_for(model_type)
    for rule in rules:
        required = rule.value
        value = metrics.get(rule.name)
        if value is None:
            _fail(
                result,
                f"metric_missing_or_invalid:{rule.name}",
                f"{rule.name} is absent from the evaluation evidence, so the rule "
                f"'{_rule_phrase(rule)}' cannot be checked",
                got=None,
                required=required,
            )
            continue
        ok = value >= required if rule.direction == "min" else value <= required
        result.checks.append(_check(rule.name, ok, value, required))
        if not ok:
            result.failed_rules.append({
                "rule": rule.name,
                "message": _failure_message(rule, value),
                "got": value,
                "required": required,
                "direction": rule.direction,
                "source": rule.source,
                "category": "metric",
            })

    result.passed = not result.failed_rules
    result.verdict = VERDICT_PASS if result.passed else VERDICT_HOLD
    if not result.passed and result.reason is None:
        result.reason = _hold_reason(result.failed_rules)
    return result


def _rule_phrase(rule: MetricRule) -> str:
    if rule.name == "sample_count":
        return f"minimum sample count {rule.value:g}"
    if rule.direction == "min":
        return f"{rule.name} at least {rule.value:g}"
    return f"{rule.name} at most {rule.value:g}"


def _failure_message(rule: MetricRule, value: float) -> str:
    if rule.name == "sample_count":
        return f"evaluation sample count {value:g} is below the required minimum {rule.value:g}"
    if rule.direction == "min":
        return f"{rule.name} {value:g} is below the required minimum {rule.value:g}"
    return f"{rule.name} {value:g} is above the allowed maximum {rule.value:g}"


def _hold_reason(failed_rules: list[dict]) -> str:
    names = [rule["rule"] for rule in failed_rules]
    return "HOLD: " + "; ".join(names)


def _evidence_ref(evidence: Mapping | None) -> dict | None:
    """The identity of the evidence, never its bulk."""
    if not isinstance(evidence, Mapping):
        return None
    metrics = evidence.get("metrics") or {}
    return {
        "schema_version": evidence.get("schema_version"),
        "fingerprint": evidence.get("fingerprint"),
        "model_name": (evidence.get("model") or {}).get("name"),
        "model_version": (evidence.get("model") or {}).get("version"),
        "dataset": (evidence.get("dataset") or {}).get("name"),
        "split": (evidence.get("dataset") or {}).get("split"),
        "sample_count": metrics.get("sample_count") if isinstance(metrics, Mapping) else None,
        "threshold": evidence.get("threshold"),
        "evaluation_time": evidence.get("evaluation_time"),
    }


__all__ = [
    "EVIDENCE_METRICS",
    "SCHEMA_VERSION",
    "STRUCTURAL_REQUIREMENTS",
    "VERDICT_HOLD",
    "VERDICT_NOT_ENFORCED",
    "VERDICT_PASS",
    "Enforcement",
    "QualityGatePolicy",
    "QualityGatePolicyError",
    "QualityGateResult",
    "evaluate_quality_gate",
    "evidence_metrics",
    "evidence_problems",
    "get_quality_gate_policy",
    "load_quality_gate_policy",
    "quality_gate_policy_path",
    "reset_quality_gate_policy_cache",
    "safe_get_quality_gate_policy",
]
