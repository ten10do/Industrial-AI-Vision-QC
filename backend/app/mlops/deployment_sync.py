"""Registry -> runtime deployment sync.

A PRODUCTION row in the database is an *intent*. It changes nothing about the
process that actually judges inspections. This module makes the gap explicit
and closable:

* ``runtime_sync`` compares, per channel, the registry PRODUCTION row against
  the deployment manifest and against the live inference service. It never
  reports IN_SYNC on absence of evidence: an unreachable inference service
  yields UNVERIFIED.
* ``apply_activation`` is the only path that writes that intent into the
  manifest the inference service pins to, plus the runtime env file. It
  re-hashes the artifact before writing, because the manifest is the last
  place a substituted artifact could still slip through.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings, resolve_path
from .attestation import sha256_file, verify_artifact_hash
from .manifest import ACTIVATED_MANIFEST_NAME, load_manifest, resolve_manifest_path

logger = logging.getLogger(__name__)

CHANNELS = ("yolo", "patchcore")

STATUS_IN_SYNC = "IN_SYNC"
STATUS_DRIFT = "DRIFT"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_NO_PRODUCTION = "NO_PRODUCTION"


class ActivationError(RuntimeError):
    pass


@dataclass
class ChannelStatus:
    channel: str
    status: str
    detail: str
    registry: dict | None = None
    manifest: dict | None = None

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "status": self.status,
            "detail": self.detail,
            "registry": self.registry,
            "manifest": self.manifest,
        }


@dataclass
class RuntimeSyncReport:
    checked_at: str
    channels: list[ChannelStatus] = field(default_factory=list)
    manifest: dict | None = None
    inference_service: dict | None = None
    overall: str = STATUS_UNVERIFIED

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "overall": self.overall,
            "manifest": self.manifest,
            "inference_service": self.inference_service,
            "channels": [c.to_dict() for c in self.channels],
        }


def _identity(row: Any) -> dict:
    return {
        "id": str(row.id),
        "model_name": row.model_name,
        "model_version": row.model_version,
        "artifact_uri": row.artifact_uri,
        "artifact_sha256": row.artifact_sha256,
        "activated_at": row.activated_at.isoformat() if row.activated_at else None,
    }


def compare_channel(channel: str, row: Any, manifest: dict) -> ChannelStatus:
    section = manifest.get(channel) or {}
    manifest_identity = {
        "model_name": section.get("model"),
        "model_version": section.get("version"),
        "artifact_sha256": section.get("sha256"),
    }
    if row is None:
        return ChannelStatus(channel, STATUS_NO_PRODUCTION,
                             "registry has no PRODUCTION model for this channel",
                             None, manifest_identity)

    drift: list[str] = []
    if section.get("model") != row.model_name:
        drift.append(f"model {section.get('model')!r} != {row.model_name!r}")
    if str(section.get("version")) != str(row.model_version):
        drift.append(f"version {section.get('version')!r} != {row.model_version!r}")
    if row.artifact_sha256 and section.get("sha256") != row.artifact_sha256:
        drift.append("artifact sha256 differs")
    if drift:
        return ChannelStatus(channel, STATUS_DRIFT,
                             "registry PRODUCTION is not what the manifest pins: " + "; ".join(drift),
                             _identity(row), manifest_identity)
    return ChannelStatus(channel, STATUS_UNVERIFIED,
                         "manifest matches registry PRODUCTION; live process not confirmed",
                         _identity(row), manifest_identity)


async def probe_inference(base_url: str | None = None, timeout: float | None = None) -> dict:
    """Best-effort probe of the inference service. Never raises."""
    settings = get_settings()
    url = (base_url or settings.inference_service_url).rstrip("/")
    limit = timeout if timeout is not None else min(settings.inference_timeout_seconds, 10.0)
    probe: dict[str, Any] = {"url": url, "reachable": False, "ready": False,
                             "deployment_version": None, "problems": [], "error": None}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=limit) as client:
            r = await client.get(f"{url}/ready")
    except Exception as exc:  # noqa: BLE001 - a probe must never break the report
        probe["error"] = f"{type(exc).__name__}: {exc}"
        return probe
    probe["reachable"] = True
    if r.status_code != 200:
        probe["error"] = f"HTTP {r.status_code}"
        return probe
    try:
        body = r.json()
    except ValueError:
        probe["error"] = "non-JSON /ready response"
        return probe
    probe["ready"] = body.get("status") == "ready"
    probe["deployment_version"] = body.get("deployment_version")
    probe["problems"] = list(body.get("problems") or [])
    return probe


def _overall(channels: list[ChannelStatus], probe: dict, manifest_version: str | None) -> str:
    statuses = [c.status for c in channels]
    if STATUS_DRIFT in statuses:
        return STATUS_DRIFT
    if not probe.get("reachable") or not probe.get("ready"):
        return STATUS_UNVERIFIED
    if manifest_version and probe.get("deployment_version") != manifest_version:
        return STATUS_DRIFT
    if any(s == STATUS_NO_PRODUCTION for s in statuses):
        return STATUS_UNVERIFIED
    return STATUS_IN_SYNC


async def runtime_sync(production_rows: dict[str, Any], manifest_path: Path | None = None) -> RuntimeSyncReport:
    path = resolve_manifest_path(manifest_path)
    report = RuntimeSyncReport(checked_at=datetime.now(timezone.utc).isoformat())
    try:
        manifest = load_manifest(path)
    except Exception as exc:  # noqa: BLE001
        report.manifest = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        report.channels = [ChannelStatus(c, STATUS_UNVERIFIED, "manifest unreadable") for c in CHANNELS]
        report.inference_service = {"reachable": False, "error": "manifest unreadable"}
        report.overall = STATUS_UNVERIFIED
        return report

    try:
        manifest_digest = sha256_file(path)
    except OSError:
        manifest_digest = None
    report.manifest = {
        "path": str(path),
        "vision_stack_version": manifest.get("vision_stack_version"),
        "sha256": manifest_digest,
        "generated": path.name == ACTIVATED_MANIFEST_NAME,
    }
    report.channels = [compare_channel(c, production_rows.get(c), manifest) for c in CHANNELS]
    report.inference_service = await probe_inference()
    report.overall = _overall(report.channels, report.inference_service, manifest.get("vision_stack_version"))
    return report


# ---- activation ----

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".activation-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _env_path() -> Path:
    return resolve_path(get_settings().runtime_env_file)


def apply_activation(
    row: Any,
    *,
    new_stack_version: str,
    manifest_path: Path | None = None,
    env_file: Path | None = None,
) -> dict:
    """Pin a PRODUCTION row into the running deployment manifest + env file.

    Re-verifies the artifact hash first: the manifest is the last artifact the
    inference service trusts, so it must be written from a verified hash.
    Requires an explicit target stack version; nothing is auto-incremented.
    """
    import yaml

    if row.status != "PRODUCTION":
        raise ActivationError(f"only a PRODUCTION model can be activated (status={row.status})")
    if not new_stack_version or not str(new_stack_version).strip():
        raise ActivationError("new_stack_version is required")

    check = verify_artifact_hash(row.artifact_uri, row.artifact_sha256)
    if not check.verified:
        raise ActivationError(f"artifact verification failed before activation: {check.detail}")

    source = resolve_manifest_path(manifest_path)
    manifest = load_manifest(source)
    target = source.with_name(ACTIVATED_MANIFEST_NAME) if source.name != ACTIVATED_MANIFEST_NAME else source

    metrics = row.metadata_json or {}
    section = {
        "model": row.model_name,
        "version": str(row.model_version),
        "model_type": row.model_type,
        "artifact_uri": row.artifact_uri,
        "sha256": row.artifact_sha256,
        "dataset_version": row.dataset_version,
        "training_run_id": row.training_run_id,
        "metrics": metrics,
        "domain": (row.domain_evidence or {}).get("domain"),
        "steel_domain_validated": bool(row.domain_validated and row.domain_evidence_verified),
        "registry_id": str(row.id),
        "activated_from": row.approved_by,
    }
    manifest[row.model_type] = {**(manifest.get(row.model_type) or {}), **section}
    manifest["vision_stack_version"] = str(new_stack_version)
    header = (
        "# GENERATED by POST /api/v1/models/{id}/activate. Do not hand-edit.\n"
        f"# source: {source.name}\n"
        f"# activated_at: {datetime.now(timezone.utc).isoformat()}\n"
    )
    _atomic_write(target, header + yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True))

    env_target = Path(env_file) if env_file else _env_path()
    env_key = "IVQC_WEIGHTS" if row.model_type == "yolo" else "IVQC_PATCHCORE_BANK"
    artifact_abs = row.artifact_uri
    from .attestation import resolve_artifact

    resolved = resolve_artifact(artifact_abs)
    env_lines = [
        f"IVQC_MANIFEST={target.as_posix()}",
        f"{env_key}={(resolved or Path(str(artifact_abs))).as_posix()}",
        f"IVQC_MODEL_VERSION={row.model_version}",
    ]
    _atomic_write(env_target, "\n".join(env_lines) + "\n")

    logger.info("activated %s@%s into %s (stack %s)", row.model_name, row.model_version, target, new_stack_version)
    return {
        "manifest_path": target.as_posix(),
        "manifest_sha256": sha256_file(target),
        "env_file": env_target.as_posix(),
        "env_keys": [line.split("=", 1)[0] for line in env_lines],
        "vision_stack_version": str(new_stack_version),
        "artifact_sha256": row.artifact_sha256,
        # The running process still holds the old weights until it is restarted
        # with this manifest. Reporting a restart requirement is the honest
        # answer; the process is not hot-swapped.
        "requires_restart": True,
        "runtime_status": STATUS_UNVERIFIED,
    }


def activation_env_snapshot() -> dict:
    env_target = _env_path()
    if not env_target.exists():
        return {"path": env_target.as_posix(), "exists": False, "values": {}}
    values: dict[str, str] = {}
    for line in env_target.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return {"path": env_target.as_posix(), "exists": True, "values": values}


def to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
