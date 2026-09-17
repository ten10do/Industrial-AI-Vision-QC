"""Fail-closed release-package validation for the immutable D3 1.3 candidate."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from steel_patchcore.candidate_registry import CandidateRegistryError, canonical_sha256, sha256_file
from steel_patchcore.dual_candidate_registry import DualCandidateRegistry

RELEASE_SCHEMA_VERSION = "steel_patchcore_d3_release_manifest_v1"
DEPENDENCY_LOCK_SCHEMA_VERSION = "steel_patchcore_d3_dependency_lock_v2"
RELEASE_REPORT_SCHEMA_VERSION = "steel_patchcore_d3_release_readiness_v1"
RELEASE_NAME = "steel-patchcore-d3-release"
RELEASE_VERSION = "1.3.0"
CANDIDATE_VERSION = "1.3.0-candidate.1"
FROZEN_THRESHOLD = 0.8471092581748962


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_hashed_requirements_lock(text: str) -> dict[str, str]:
    """Return normalized pins and fail closed when any requirement lacks a SHA-256."""
    pins: dict[str, str] = {}
    current: str | None = None
    hash_count = 0
    for raw in text.splitlines():
        line = raw.strip()
        requirement = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s+\\)?", line)
        if requirement:
            if current is not None and hash_count == 0:
                raise CandidateRegistryError(f"RELEASE_INSTALL_LOCK_HASH_MISSING:{current}")
            current = re.sub(r"[-_.]+", "-", requirement.group(1)).lower()
            if current in pins:
                raise CandidateRegistryError(f"RELEASE_INSTALL_LOCK_DUPLICATE:{current}")
            pins[current] = requirement.group(2)
            hash_count = 0
            continue
        if line.startswith("--hash="):
            if current is None or re.fullmatch(r"--hash=sha256:[0-9a-f]{64}(?:\s+\\)?", line) is None:
                raise CandidateRegistryError("RELEASE_INSTALL_LOCK_HASH_INVALID")
            hash_count += 1
            continue
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        raise CandidateRegistryError(f"RELEASE_INSTALL_LOCK_LINE_INVALID:{line}")
    if current is None or hash_count == 0:
        raise CandidateRegistryError("RELEASE_INSTALL_LOCK_EMPTY_OR_UNHASHED")
    return pins


def validate_dependency_lock(lock: Mapping) -> None:
    required = {
        "schema_version", "python", "cuda_wheel_index", "requirement_files",
        "declared_packages", "qualification_runtime", "install", "security_audit",
        "lock_payload_sha256",
    }
    if set(lock) != required or lock.get("schema_version") != DEPENDENCY_LOCK_SCHEMA_VERSION:
        raise CandidateRegistryError("RELEASE_DEPENDENCY_LOCK_SCHEMA_MISMATCH")
    if lock.get("python") != "3.11" or lock.get("cuda_wheel_index") != "https://download.pytorch.org/whl/cu130":
        raise CandidateRegistryError("RELEASE_DEPENDENCY_RUNTIME_MISMATCH")
    requirement_files = lock.get("requirement_files", {})
    if set(requirement_files) != {
        "inference", "training", "backend", "vision_contract",
        "qualified_runtime_input", "qualified_runtime_lock",
    }:
        raise CandidateRegistryError("RELEASE_REQUIREMENT_FILE_SET_MISMATCH")
    if not all(_is_sha256(row.get("sha256")) and isinstance(row.get("uri"), str) for row in requirement_files.values()):
        raise CandidateRegistryError("RELEASE_REQUIREMENT_HASH_INVALID")
    packages = lock.get("declared_packages", {})
    if set(packages) != {"inference", "training", "backend"} or not all(isinstance(rows, list) and rows for rows in packages.values()):
        raise CandidateRegistryError("RELEASE_DECLARED_PACKAGES_INVALID")
    if "torch==2.13.0+cu130" not in packages["inference"] or "torch==2.13.0+cu130" not in packages["training"]:
        raise CandidateRegistryError("RELEASE_TORCH_LOCK_MISMATCH")
    install = lock.get("install", {})
    if install != {
        "requirements_uri": requirement_files["qualified_runtime_lock"]["uri"],
        "require_hashes": True,
        "python_tag": "cp311",
        "platform_tag": "win_amd64",
    }:
        raise CandidateRegistryError("RELEASE_INSTALL_LOCK_METADATA_INVALID")
    audit = lock.get("security_audit", {})
    if set(audit) != {"uri", "sha256", "cuda_local_version_normalization"} or not _is_sha256(audit.get("sha256")):
        raise CandidateRegistryError("RELEASE_SECURITY_AUDIT_REFERENCE_INVALID")
    if audit.get("cuda_local_version_normalization") != {
        "torch": "2.13.0", "torchvision": "0.28.0",
    }:
        raise CandidateRegistryError("RELEASE_SECURITY_AUDIT_NORMALIZATION_INVALID")
    payload = dict(lock)
    recorded = payload.pop("lock_payload_sha256", None)
    if not _is_sha256(recorded) or canonical_sha256(payload) != recorded:
        raise CandidateRegistryError("RELEASE_DEPENDENCY_LOCK_PAYLOAD_MISMATCH")


def validate_release_manifest(manifest: Mapping) -> None:
    required = {
        "schema_version", "status", "release_name", "release_version", "created_at",
        "candidate", "threshold", "artifact_hashes", "protocol_versions",
        "dependency_lock", "qualification_evidence", "production_promotion",
        "manifest_payload_sha256",
    }
    if set(manifest) != required or manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise CandidateRegistryError("RELEASE_MANIFEST_SCHEMA_MISMATCH")
    if manifest.get("status") != "RELEASE_CANDIDATE_PACKAGE" or manifest.get("production_promotion") is not False:
        raise CandidateRegistryError("RELEASE_CANDIDATE_ONLY_REQUIRED")
    if manifest.get("release_name") != RELEASE_NAME or manifest.get("release_version") != RELEASE_VERSION:
        raise CandidateRegistryError("RELEASE_IDENTITY_MISMATCH")
    candidate = manifest.get("candidate", {})
    if candidate.get("model_version") != CANDIDATE_VERSION or not _is_sha256(candidate.get("manifest_sha256")):
        raise CandidateRegistryError("RELEASE_CANDIDATE_LINEAGE_INVALID")
    if manifest.get("threshold") != FROZEN_THRESHOLD:
        raise CandidateRegistryError("RELEASE_THRESHOLD_CHANGED")
    hashes = manifest.get("artifact_hashes", {})
    required_hashes = {
        "model", "weights", "image_bank", "whitening", "localization_bank_bundle",
        "R-L1", "R-L2", "feature", "protocol",
    }
    if set(hashes) != required_hashes or not all(_is_sha256(value) for value in hashes.values()):
        raise CandidateRegistryError("RELEASE_ARTIFACT_HASH_SET_MISMATCH")
    if hashes["model"] != hashes["weights"]:
        raise CandidateRegistryError("RELEASE_MODEL_WEIGHT_HASH_MISMATCH")
    protocols = manifest.get("protocol_versions", {})
    if set(protocols) != {"candidate_manifest", "image_branch", "localization_branch", "release_manifest", "dependency_lock"}:
        raise CandidateRegistryError("RELEASE_PROTOCOL_SET_MISMATCH")
    dependency = manifest.get("dependency_lock", {})
    if set(dependency) != {"uri", "sha256"} or not _is_sha256(dependency.get("sha256")):
        raise CandidateRegistryError("RELEASE_DEPENDENCY_REFERENCE_INVALID")
    evidence = manifest.get("qualification_evidence", {})
    expected = {
        "dual_branch": "PASS",
        "production_readiness": "PRODUCTION_CANDIDATE_QUALIFIED",
        "factory_acceptance": "FACTORY_ACCEPTANCE_PASS",
    }
    if set(evidence) != set(expected):
        raise CandidateRegistryError("RELEASE_QUALIFICATION_SET_MISMATCH")
    for name, verdict in expected.items():
        row = evidence[name]
        if set(row) != {"uri", "sha256", "verdict"} or row["verdict"] != verdict or not _is_sha256(row["sha256"]):
            raise CandidateRegistryError(f"RELEASE_QUALIFICATION_INVALID:{name}")
    payload = dict(manifest)
    recorded = payload.pop("manifest_payload_sha256", None)
    if not _is_sha256(recorded) or canonical_sha256(payload) != recorded:
        raise CandidateRegistryError("RELEASE_MANIFEST_PAYLOAD_SHA256_MISMATCH")


@dataclass(frozen=True)
class LoadedReleasePackage:
    manifest: dict
    dependency_lock: dict
    artifact_hashes: dict[str, str]


class ReleasePackageRegistry:
    """Verifies a release package without registering or promoting it."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def _path(self, uri: str) -> Path:
        path = (self.project_root / uri).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise CandidateRegistryError(f"RELEASE_REFERENCE_OUTSIDE_ROOT:{uri}") from exc
        return path

    def load(self, manifest_path: str | Path) -> LoadedReleasePackage:
        path = Path(manifest_path).resolve()
        try:
            path.relative_to(self.project_root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise CandidateRegistryError("RELEASE_MANIFEST_UNREADABLE") from exc
        validate_release_manifest(manifest)
        candidate_path = self._path(manifest["candidate"]["manifest_uri"])
        if sha256_file(candidate_path) != manifest["candidate"]["manifest_sha256"]:
            raise CandidateRegistryError("RELEASE_CANDIDATE_MANIFEST_SHA_MISMATCH")
        lock_path = self._path(manifest["dependency_lock"]["uri"])
        if sha256_file(lock_path) != manifest["dependency_lock"]["sha256"]:
            raise CandidateRegistryError("RELEASE_DEPENDENCY_LOCK_SHA_MISMATCH")
        dependency_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        validate_dependency_lock(dependency_lock)
        for row in dependency_lock["requirement_files"].values():
            if sha256_file(self._path(row["uri"])) != row["sha256"]:
                raise CandidateRegistryError(f"RELEASE_REQUIREMENT_SHA_MISMATCH:{row['uri']}")
        requirement_files = dependency_lock["requirement_files"]
        install_lock_path = self._path(requirement_files["qualified_runtime_lock"]["uri"])
        install_lock_text = install_lock_path.read_text(encoding="utf-8")
        if f"--extra-index-url {dependency_lock['cuda_wheel_index']}" not in install_lock_text:
            raise CandidateRegistryError("RELEASE_CUDA_INDEX_MISSING")
        locked_pins = parse_hashed_requirements_lock(install_lock_text)
        input_text = self._path(requirement_files["qualified_runtime_input"]["uri"]).read_text(encoding="utf-8")
        input_pins = {
            re.sub(r"[-_.]+", "-", match.group(1)).lower(): match.group(2)
            for raw in input_text.splitlines()
            if (match := re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", raw.strip()))
        }
        if locked_pins != input_pins:
            raise CandidateRegistryError("RELEASE_INSTALL_LOCK_INPUT_MISMATCH")
        runtime_packages = dependency_lock["qualification_runtime"].get("packages", {})
        for name, runtime_version in runtime_packages.items():
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            if locked_pins.get(normalized) != runtime_version:
                raise CandidateRegistryError(f"RELEASE_QUALIFIED_RUNTIME_PIN_MISMATCH:{name}")
        audit_reference = dependency_lock["security_audit"]
        audit_path = self._path(audit_reference["uri"])
        if sha256_file(audit_path) != audit_reference["sha256"]:
            raise CandidateRegistryError("RELEASE_SECURITY_AUDIT_SHA_MISMATCH")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("source_lock", {}).get("sha256") != sha256_file(install_lock_path):
            raise CandidateRegistryError("RELEASE_SECURITY_AUDIT_SOURCE_MISMATCH")
        audited = {
            row.get("name"): row for row in audit.get("dependencies", [])
            if isinstance(row, Mapping)
        }
        for name, normalized_version in audit_reference["cuda_local_version_normalization"].items():
            row = audited.get(name)
            if not row or row.get("version") != normalized_version or row.get("skip_reason"):
                raise CandidateRegistryError(f"RELEASE_CUDA_DEPENDENCY_NOT_AUDITED:{name}")
        has_findings = bool(audit.get("skipped_dependencies") or audit.get("vulnerabilities"))
        if audit.get("verdict") != ("FAIL" if has_findings else "PASS"):
            raise CandidateRegistryError("RELEASE_DEPENDENCY_AUDIT_VERDICT_MISMATCH")
        for name, row in manifest["qualification_evidence"].items():
            report_path = self._path(row["uri"])
            if sha256_file(report_path) != row["sha256"]:
                raise CandidateRegistryError(f"RELEASE_QUALIFICATION_SHA_MISMATCH:{name}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("verdict") != row["verdict"]:
                raise CandidateRegistryError(f"RELEASE_QUALIFICATION_VERDICT_MISMATCH:{name}")
        candidate = DualCandidateRegistry(self.project_root).load_manifest(candidate_path)
        _, actual = DualCandidateRegistry(self.project_root).verify_artifacts(candidate)
        expected = manifest["artifact_hashes"]
        comparisons = {
            "model": candidate["hashes"]["model_sha256"],
            "weights": actual["weights"],
            "image_bank": actual["image_bank"],
            "whitening": actual["whitening"],
            "localization_bank_bundle": candidate["hashes"]["localization_bank_sha256"],
            "R-L1": actual["R-L1"],
            "R-L2": actual["R-L2"],
            "feature": candidate["hashes"]["feature_sha256"],
            "protocol": actual["protocol"],
        }
        if comparisons != expected:
            raise CandidateRegistryError("RELEASE_ARTIFACT_LINEAGE_MISMATCH")
        return LoadedReleasePackage(dict(manifest), dependency_lock, actual)


def validate_release_report(report: Mapping) -> None:
    if report.get("schema_version") != RELEASE_REPORT_SCHEMA_VERSION:
        raise CandidateRegistryError("RELEASE_REPORT_SCHEMA_MISMATCH")
    if report.get("production_promotion") is not False or report.get("automatic_retraining") is not False:
        raise CandidateRegistryError("RELEASE_REPORT_FORBIDDEN_ACTION")
    gates = report.get("gates", {})
    if set(gates) != {"manifest_freeze", "documentation", "clean_environment", "security", "tests"}:
        raise CandidateRegistryError("RELEASE_REPORT_GATE_SET_MISMATCH")
    expected = "PASS" if all(row.get("verdict") == "PASS" for row in gates.values()) else "FAIL"
    if report.get("verdict") != expected:
        raise CandidateRegistryError("RELEASE_REPORT_VERDICT_MISMATCH")
    if not isinstance(report.get("remaining_risks"), list):
        raise CandidateRegistryError("RELEASE_REPORT_RISK_SCHEMA_MISMATCH")


__all__ = [
    "CANDIDATE_VERSION", "DEPENDENCY_LOCK_SCHEMA_VERSION", "FROZEN_THRESHOLD",
    "LoadedReleasePackage", "RELEASE_NAME", "RELEASE_REPORT_SCHEMA_VERSION",
    "RELEASE_SCHEMA_VERSION", "RELEASE_VERSION", "ReleasePackageRegistry",
    "parse_hashed_requirements_lock", "validate_dependency_lock", "validate_release_manifest",
    "validate_release_report",
]
