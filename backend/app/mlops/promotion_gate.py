"""Promotion gate (8F): CANDIDATE -> PRODUCTION must pass measurable gates.

Pure functions, no DB coupling. Two rules define this gate:

1. Thresholds come from the server-owned policy file, never from the caller.
   A caller may tighten a threshold, never relax one. Sending
   ``{"mAP50": 0, "recall": 0}`` for a model that scores 0 is now a policy
   violation and blocks the promotion.
2. A metric only counts when its provenance is proven. Metrics, the domain
   verdict and the artifact hash must be attested by the signed trusted
   pipeline and independently re-verified by this server. Unverified input
   blocks the promotion instead of degrading to a warning.

In particular, a cross-domain MVTec PatchCore (steel_domain_validated=false)
can never be promoted to a steel production model even if its AUROC is
perfect, and a model with no attested evaluation can never be promoted at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gate_policy import MetricRule, PromotionPolicy, safe_get_policy

PROVENANCE_SCHEMA = "ivqc_model_provenance_v1"


@dataclass
class GateResult:
    passed: bool = False
    checks: list[dict] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    policy: dict | None = None

    def to_dict(self) -> dict:
        out = {"passed": self.passed, "checks": self.checks, "blocked": self.blocked}
        if self.policy is not None:
            out["policy"] = self.policy
        return out


def _check(name: str, ok: bool, got: Any, required: Any, category: str = "metric") -> dict:
    return {
        "check": name,
        "passed": bool(ok),
        "got": got,
        "required": required,
        "category": category,
        "blocked_by_domain": category == "domain",
    }


def default_provenance() -> dict:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "metrics_attested": False,
        "attested_by": None,
        "artifact_hash_verified": False,
        "domain_evidence_verified": False,
        "domain": None,
    }


def _coerce_metric(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return None


def _metric_checks(metrics: dict, rules: tuple[MetricRule, ...], thresholds: dict[str, float]):
    checks, blocked = [], []
    for rule in rules:
        required = thresholds[rule.name]
        value = _coerce_metric((metrics or {}).get(rule.name))
        if value is None:
            checks.append(_check(rule.name, False, (metrics or {}).get(rule.name), required))
            blocked.append(f"metric_missing_or_invalid:{rule.name}")
            continue
        ok = value >= required if rule.direction == "min" else value <= required
        checks.append(_check(rule.name, ok, value, required))
        if not ok:
            blocked.append(rule.name)
    return checks, blocked


def _provenance_checks(provenance: dict, requirements: dict[str, bool], domain_validated: bool,
                       required_domain: str) -> tuple[list[dict], list[str]]:
    checks: list[dict] = []
    blocked: list[str] = []

    def add(key: str, label: str, ok: bool) -> None:
        checks.append(_check(label, ok, provenance.get(key), True, category="provenance"))
        if not ok:
            blocked.append(f"provenance:{label}")

    if requirements.get("require_attested_metrics", True):
        add("metrics_attested", "metrics_attested", bool(provenance.get("metrics_attested")))
    if requirements.get("require_verified_artifact_hash", True):
        add("artifact_hash_verified", "artifact_hash_verified", bool(provenance.get("artifact_hash_verified")))
    if requirements.get("require_verified_domain_evidence", True):
        add("domain_evidence_verified", "domain_evidence_verified", bool(provenance.get("domain_evidence_verified")))

    evidence_domain = provenance.get("domain")
    domain_matches = evidence_domain is None or str(evidence_domain) == required_domain
    checks.append(_check("domain_match", domain_matches, evidence_domain, required_domain, category="domain"))
    if not domain_matches:
        blocked.append(f"domain_mismatch:{evidence_domain}!={required_domain}")

    checks.append(_check("domain", domain_validated, domain_validated, True, category="domain"))
    if not domain_validated:
        blocked.append(f"domain not validated for {required_domain}")

    return checks, blocked


def evaluate(
    model_type: str,
    *,
    metrics: dict,
    thresholds: dict | None = None,
    domain_validated: bool,
    required_domain: str,
    provenance: dict | None = None,
    policy: PromotionPolicy | None = None,
) -> GateResult:
    """Evaluate the promotion gate.

    `thresholds` is accepted only for tightening: every value must be at least
    as strict as the policy value for that metric. Unknown or relaxing values
    are violations and block the promotion.
    """
    result = GateResult()

    if policy is None:
        policy, policy_error = safe_get_policy()
        if policy is None:
            result.blocked.append(f"policy_unavailable: {policy_error}")
            result.checks.append(_check("policy", False, policy_error, "loadable policy", category="policy"))
            result.passed = False
            return result

    if model_type not in policy.model_types:
        result.blocked.append(f"unknown model_type {model_type}")
        result.checks.append(_check("model_type", False, model_type, sorted(policy.model_types), category="policy"))
        result.passed = False
        return result

    rules = policy.rules_for(model_type)
    effective, violations = policy.resolve_thresholds(model_type, thresholds)
    result.policy = policy.to_dict(model_type, effective)
    result.checks.append(
        _check("threshold_policy", not violations, thresholds or {}, effective, category="policy")
    )
    result.blocked.extend(violations)

    metric_checks, metric_blocked = _metric_checks(metrics, rules, effective)
    result.checks.extend(metric_checks)
    result.blocked.extend(metric_blocked)

    prov = {**default_provenance(), **(provenance or {})}
    prov_checks, prov_blocked = _provenance_checks(
        prov, policy.provenance, domain_validated, required_domain
    )
    result.checks.extend(prov_checks)
    result.blocked.extend(prov_blocked)

    result.passed = not result.blocked
    return result
