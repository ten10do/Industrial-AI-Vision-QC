"""Deployment manifest (8D / 8E).

A fixed manifest pins the whole AI stack that judges every inspection:

    vision_stack_version: "2026.08.1"
    yolo:        {model, version, sha256}
    patchcore:   {model, version, sha256}
    fusion:      {version}
    quality_rules: {version}

The backend stamps inspections.deployment_version from it, and the
inference service resolves + SHA256-validates artifacts against it before
declaring READY. No version-less `best.pt` startup (8C boundary).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "deployment_manifest.yaml"

# Written by POST /api/v1/models/{id}/activate. The hand-maintained manifest
# stays the source of truth for humans; the generated one is what the running
# stack is pinned to, and it wins when present so that a registry PRODUCTION
# change actually reaches the inference process instead of staying in the DB.
ACTIVATED_MANIFEST_NAME = "deployment_manifest.activated.yaml"

_MANIFEST: dict[str, Any] | None = None


def resolve_manifest_path(path: Path | None = None) -> Path:
    """Precedence: explicit argument > IVQC_MANIFEST > activated > default."""
    if path is not None:
        return Path(path)
    env = os.environ.get("IVQC_MANIFEST")
    if env:
        return Path(env).resolve()
    activated = DEFAULT_MANIFEST_PATH.with_name(ACTIVATED_MANIFEST_NAME)
    return activated if activated.exists() else DEFAULT_MANIFEST_PATH


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the deployment manifest YAML."""
    import yaml

    p = resolve_manifest_path(path)
    if not p.exists():
        raise FileNotFoundError(f"deployment manifest not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "vision_stack_version" not in data:
        raise ValueError("manifest must contain vision_stack_version")
    for section in ("yolo", "patchcore"):
        if section not in data:
            raise ValueError(f"manifest missing section {section}")
        for key in ("model", "version"):
            if key not in data[section]:
                raise ValueError(f"manifest {section}.{key} missing")
    return data


def get_manifest(path: Path | None = None) -> dict[str, Any]:
    """Cached manifest. An explicit path always re-reads (no cache pollution)."""
    global _MANIFEST
    if path is not None:
        return load_manifest(path)
    if _MANIFEST is None:
        _MANIFEST = load_manifest(resolve_manifest_path())
    return _MANIFEST


def reset_manifest_cache() -> None:
    global _MANIFEST
    _MANIFEST = None


def current_deployment_version(path: Path | None = None) -> str:
    return str(get_manifest(path)["vision_stack_version"])


def dataset_version_for_model(model_name: str | None) -> str | None:
    """Resolve the training dataset identity for a model name from the pinned
    manifest (8D/8K). Substring match (e.g. inspection.model_name="yolov8s"
    -> yolo section "neu-yolov8s" -> dataset_version "neu-det-yolo-v1").
    Returns None when the model is not part of the pinned stack."""
    if not model_name:
        return None
    m = get_manifest()
    for section in ("yolo", "patchcore"):
        sec = m[section]
        pinned = str(sec.get("model", ""))
        if model_name == pinned or model_name.lower() in pinned.lower():
            return sec.get("dataset_version")
    return None


def validate_artifacts(manifest: dict[str, Any], root: Path) -> list[str]:
    """Return a list of problems. Empty list means all artifacts pass.

    artifact_uri is interpreted relative to the project root (`root`)."""
    problems: list[str] = []
    for section in ("yolo", "patchcore"):
        m = manifest[section]
        uri = m.get("artifact_uri") or m.get("path") or m.get("model")
        if not uri:
            problems.append(f"{section}: no artifact_uri")
            continue
        p = Path(uri)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            problems.append(f"{section}: artifact missing {p}")
            continue
        expected = m.get("sha256")
        if expected:
            actual = sha256_of(p)
            if actual != expected:
                problems.append(f"{section}: sha256 mismatch for {p} (expected {expected}, got {actual[:16]}...)")
    return problems
