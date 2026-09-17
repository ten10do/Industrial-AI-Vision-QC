"""Audit the qualified CUDA lock without losing PyPI advisory coverage."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.candidate_registry import canonical_sha256, sha256_file  # noqa: E402
from steel_patchcore.d3_operational import atomic_write_json  # noqa: E402
from steel_patchcore.d3_release_package import parse_hashed_requirements_lock  # noqa: E402

LOCK = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0/requirements-cu130.lock"
DEPENDENCY_LOCK = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0/dependency-lock.json"
RELEASE_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0/manifest.json"
REPORT = ROOT / "docs/release/inference-dependency-audit.json"
CUDA_PACKAGES = {"torch", "torchvision"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def refresh_release_references() -> None:
    dependency_lock = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    dependency_lock["security_audit"]["sha256"] = sha256_file(REPORT)
    dependency_payload = dict(dependency_lock)
    dependency_payload.pop("lock_payload_sha256", None)
    dependency_lock["lock_payload_sha256"] = canonical_sha256(dependency_payload)
    atomic_write_json(DEPENDENCY_LOCK, dependency_lock)

    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    manifest["dependency_lock"]["sha256"] = sha256_file(DEPENDENCY_LOCK)
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest_payload)
    atomic_write_json(RELEASE_MANIFEST, manifest)


def audit() -> dict:
    pins = parse_hashed_requirements_lock(LOCK.read_text(encoding="utf-8"))
    normalized = {
        name: version.partition("+")[0] if name in CUDA_PACKAGES else version
        for name, version in pins.items()
    }
    pip_audit = shutil.which("pip-audit")
    if pip_audit is None:
        raise RuntimeError("pip-audit is required to generate the release dependency audit")
    with tempfile.TemporaryDirectory(prefix="steel-d3-audit-") as directory:
        requirements = Path(directory) / "requirements-audit.txt"
        raw_report = Path(directory) / "pip-audit.json"
        requirements.write_text(
            "\n".join(f"{name}=={version}" for name, version in sorted(normalized.items())) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [pip_audit, "--requirement", str(requirements), "--no-deps", "--disable-pip",
             "--format", "json", "--output", str(raw_report)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in {0, 1} or not raw_report.is_file():
            raise RuntimeError(f"pip-audit failed ({result.returncode}): {result.stderr.strip()}")
        payload = json.loads(raw_report.read_text(encoding="utf-8"))
    dependencies = payload.get("dependencies", [])
    skipped = [row for row in dependencies if row.get("skip_reason")]
    vulnerabilities_by_key = {}
    for row in dependencies:
        for finding in row.get("vulns", []):
            key = (row["name"], row.get("version"), finding["id"])
            vulnerabilities_by_key.setdefault(
                key, {"package": row["name"], "version": row.get("version"), **finding}
            )
    vulnerabilities = list(vulnerabilities_by_key.values())
    audited = {row.get("name"): row for row in dependencies}
    for name in CUDA_PACKAGES:
        row = audited.get(name)
        if not row or row.get("version") != normalized[name] or row.get("skip_reason"):
            raise RuntimeError(f"CUDA dependency was not audited: {name}")
    report = {
        "schema_version": "steel_patchcore_d3_dependency_audit_v1",
        "source_lock": {
            "uri": LOCK.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(LOCK),
        },
        "cuda_local_version_normalization": {
            name: {"locked": pins[name], "audited": normalized[name]}
            for name in sorted(CUDA_PACKAGES)
        },
        "dependencies": dependencies,
        "fixes": payload.get("fixes", []),
        "skipped_dependencies": skipped,
        "vulnerabilities": vulnerabilities,
        "verdict": "PASS" if not skipped and not vulnerabilities else "FAIL",
        "generated_at": utc_now(),
    }
    atomic_write_json(REPORT, report)
    refresh_release_references()
    return report


if __name__ == "__main__":
    raise SystemExit(0 if audit()["verdict"] == "PASS" else 1)
