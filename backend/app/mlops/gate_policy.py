"""Server-owned promotion thresholds.

The acceptance criteria live in a file the process reads, never in a request
body. Loading is fail-closed: a missing, malformed, out-of-bound or
hash-mismatched policy disables promotion entirely rather than falling back
to a permissive default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import get_settings, resolve_path

SCHEMA_VERSION = "ivqc_promotion_policy_v1"


class PolicyError(RuntimeError):
    """Raised when the policy cannot be trusted. Callers must fail closed."""


@dataclass(frozen=True)
class MetricRule:
    name: str
    direction: str  # "min" (higher is better) | "max" (lower is better)
    value: float
    bound: float
    # Where the number came from. The promotion policy leaves it unset; the
    # evaluation quality gate requires it, because a threshold nobody can
    # trace is a threshold nobody can defend in an audit.
    source: str | None = None

    def compare(self, got: float) -> bool:
        return got >= self.value if self.direction == "min" else got <= self.value

    def is_tightening(self, override: float) -> bool:
        return override >= self.value if self.direction == "min" else override <= self.value


@dataclass(frozen=True)
class ModelTypePolicy:
    rules: tuple[MetricRule, ...]

    def thresholds(self) -> dict[str, float]:
        return {r.name: r.value for r in self.rules}


def resolve_rules(
    rules: tuple[MetricRule, ...], overrides: dict | None
) -> tuple[dict[str, float], list[str]]:
    """Return (effective thresholds, violations) for a rule set.

    An override may only tighten a threshold. Anything else, including an
    unknown metric name, is a violation and blocks the promotion. Shared by
    the promotion policy and the evaluation quality gate so that the two can
    never drift into different meanings of "tighter".
    """
    effective = {r.name: r.value for r in rules}
    violations: list[str] = []
    for key, raw in (overrides or {}).items():
        rule = next((r for r in rules if r.name == key), None)
        if rule is None:
            violations.append(f"unknown_threshold:{key}")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            violations.append(f"non_numeric_threshold:{key}")
            continue
        if value != value or value in (float("inf"), float("-inf")):
            violations.append(f"non_finite_threshold:{key}")
            continue
        if not rule.is_tightening(value):
            violations.append(
                f"threshold_relaxation_forbidden:{key}:{value}<{rule.value}"
                if rule.direction == "min"
                else f"threshold_relaxation_forbidden:{key}:{value}>{rule.value}"
            )
            continue
        effective[key] = value
    return effective, violations


@dataclass(frozen=True)
class PromotionPolicy:
    policy_id: str
    sha256: str
    pinned: bool
    model_types: dict[str, ModelTypePolicy]
    provenance: dict[str, bool]
    approval: dict[str, Any]

    def rules_for(self, model_type: str) -> tuple[MetricRule, ...]:
        return self.model_types[model_type].rules

    def resolve_thresholds(self, model_type: str, overrides: dict | None) -> tuple[dict[str, float], list[str]]:
        """Return (effective thresholds, violations).

        An override may only tighten a threshold. Anything else, including an
        unknown metric name, is a violation and blocks the promotion.
        """
        return resolve_rules(self.rules_for(model_type), overrides)

    def to_dict(self, model_type: str, effective: dict[str, float]) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_sha256": self.sha256,
            "policy_pinned": self.pinned,
            "policy_path": str(policy_path()),
            "thresholds_used": dict(effective),
            "provenance_requirements": dict(self.provenance),
        }


def policy_path() -> Path:
    return resolve_path(get_settings().promotion_policy_path)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_rule(name: str, raw: Any) -> MetricRule:
    if not isinstance(raw, dict):
        raise PolicyError(f"policy metric {name}: expected a mapping")
    direction = raw.get("direction")
    if direction not in ("min", "max"):
        raise PolicyError(f"policy metric {name}: direction must be min or max")
    try:
        value = float(raw["value"])
        bound = float(raw["bound"])
    except KeyError as exc:
        raise PolicyError(f"policy metric {name}: missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"policy metric {name}: value/bound must be numeric") from exc
    if direction == "min" and value < bound:
        raise PolicyError(f"policy metric {name}: value {value} below hard floor {bound}")
    if direction == "max" and value > bound:
        raise PolicyError(f"policy metric {name}: value {value} above hard ceiling {bound}")
    return MetricRule(name=name, direction=direction, value=value, bound=bound)


def load_policy(path: Path | None = None) -> PromotionPolicy:
    import yaml

    p = path or policy_path()
    if not p.exists():
        raise PolicyError(f"promotion policy not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is fatal
        raise PolicyError(f"promotion policy unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError("promotion policy must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise PolicyError(f"promotion policy schema mismatch: {data.get('schema_version')!r}")

    digest = sha256_of_file(p)
    expected = (get_settings().promotion_policy_sha256 or "").strip().lower()
    if expected and expected != digest:
        raise PolicyError(
            f"promotion policy hash pin mismatch: expected {expected}, got {digest}. "
            "Refusing to evaluate any promotion against an unverified policy."
        )

    raw_thresholds = data.get("thresholds")
    if not isinstance(raw_thresholds, dict) or not raw_thresholds:
        raise PolicyError("promotion policy has no thresholds section")
    model_types: dict[str, ModelTypePolicy] = {}
    for model_type, metrics in raw_thresholds.items():
        if not isinstance(metrics, dict) or not metrics:
            raise PolicyError(f"promotion policy: {model_type} has no metrics")
        model_types[str(model_type)] = ModelTypePolicy(
            rules=tuple(_parse_rule(str(k), v) for k, v in metrics.items())
        )

    provenance = data.get("provenance_requirements") or {}
    if not isinstance(provenance, dict):
        raise PolicyError("promotion policy: provenance_requirements must be a mapping")
    approval = data.get("approval") or {}
    if not isinstance(approval, dict):
        raise PolicyError("promotion policy: approval must be a mapping")

    return PromotionPolicy(
        policy_id=str(data.get("policy_id") or "unidentified-policy"),
        sha256=digest,
        pinned=bool(expected),
        model_types=model_types,
        provenance={str(k): bool(v) for k, v in provenance.items()},
        approval=approval,
    )


_POLICY: PromotionPolicy | None = None


def get_policy(path: Path | None = None) -> PromotionPolicy:
    """Cached policy. Raises PolicyError when the file cannot be trusted."""
    global _POLICY
    if _POLICY is None:
        _POLICY = load_policy(path)
    return _POLICY


def reset_policy_cache() -> None:
    global _POLICY
    _POLICY = None


def safe_get_policy(path: Path | None = None) -> tuple[PromotionPolicy | None, str | None]:
    """Never raises. Returns (policy, error). Used where the caller must
    convert a policy failure into a blocked gate instead of a 500."""
    try:
        return get_policy(path), None
    except PolicyError as exc:
        return None, str(exc)
