"""D3 release package freeze, isolation and readiness tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.candidate_registry import CandidateRegistryError, canonical_sha256  # noqa: E402
from steel_patchcore.d3_release_package import (  # noqa: E402
    ReleasePackageRegistry,
    parse_hashed_requirements_lock,
    validate_dependency_lock,
    validate_release_manifest,
    validate_release_report,
)

RELEASE_DIR = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0"
MANIFEST = RELEASE_DIR / "manifest.json"
LOCK = RELEASE_DIR / "dependency-lock.json"
INSTALL_LOCK = RELEASE_DIR / "requirements-cu130.lock"
DEPENDENCY_AUDIT = ROOT / "docs/release/inference-dependency-audit.json"


def test_dependency_lock_is_canonical_and_cuda_pinned():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_dependency_lock(lock)
    assert lock["python"] == "3.11"
    assert lock["cuda_wheel_index"].endswith("/cu130")
    assert "torch==2.13.0+cu130" in lock["declared_packages"]["inference"]
    assert lock["install"]["require_hashes"] is True
    assert lock["requirement_files"]["qualified_runtime_lock"]["uri"].endswith("requirements-cu130.lock")


def test_qualified_runtime_lock_is_fully_hashed_and_matches_evidence():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    pins = parse_hashed_requirements_lock(INSTALL_LOCK.read_text(encoding="utf-8"))
    assert pins["torch"] == "2.13.0+cu130"
    assert pins["torchvision"] == "0.28.0+cu130"
    assert pins["numpy"] == lock["qualification_runtime"]["packages"]["numpy"]
    assert pins["pandas"] == lock["qualification_runtime"]["packages"]["pandas"]


def test_unhashed_cuda_wheel_fails_closed():
    text = INSTALL_LOCK.read_text(encoding="utf-8")
    tampered = text.replace(
        "    --hash=sha256:45e97bd9bc0416f4f4190b5098c55119a389fa5a7c8bbf2639f08f1d04e0a0dc\n",
        "",
    )
    with pytest.raises(CandidateRegistryError, match="RELEASE_INSTALL_LOCK_HASH_MISSING:torch"):
        parse_hashed_requirements_lock(tampered)


def test_cuda_wheels_are_audited_as_upstream_versions_without_skip():
    audit = json.loads(DEPENDENCY_AUDIT.read_text(encoding="utf-8"))
    dependencies = {row["name"]: row for row in audit["dependencies"]}
    assert dependencies["torch"]["version"] == "2.13.0"
    assert dependencies["torchvision"]["version"] == "0.28.0"
    assert "skip_reason" not in dependencies["torch"]
    assert "skip_reason" not in dependencies["torchvision"]
    assert audit["skipped_dependencies"] == []


def test_release_manifest_freezes_candidate_without_promotion():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_release_manifest(manifest)
    assert manifest["status"] == "RELEASE_CANDIDATE_PACKAGE"
    assert manifest["candidate"]["model_version"] == "1.3.0-candidate.1"
    assert manifest["threshold"] == 0.8471092581748962
    assert manifest["production_promotion"] is False


def test_release_registry_verifies_all_lineage_and_artifacts():
    package = ReleasePackageRegistry(ROOT).load(MANIFEST)
    assert set(package.artifact_hashes) == {
        "legacy_manifest", "weights", "whitening", "image_bank", "R-L1", "R-L2", "protocol", "investigation_results"
    }


def test_release_manifest_tamper_fails_closed():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(manifest)
    tampered["threshold"] += 0.001
    payload = dict(tampered)
    payload.pop("manifest_payload_sha256")
    tampered["manifest_payload_sha256"] = canonical_sha256(payload)
    with pytest.raises(CandidateRegistryError, match="RELEASE_THRESHOLD_CHANGED"):
        validate_release_manifest(tampered)


def test_release_report_is_derived_and_candidate_only():
    report = {
        "schema_version": "steel_patchcore_d3_release_readiness_v1",
        "candidate_status": "PRODUCTION_CANDIDATE_QUALIFIED",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "gates": {name: {"verdict": "PASS"} for name in (
            "manifest_freeze", "documentation", "clean_environment", "security", "tests"
        )},
        "verdict": "PASS",
        "remaining_risks": [],
        "production_promotion": False,
        "automatic_retraining": False,
    }
    validate_release_report(report)
    assert report["verdict"] == "PASS"
    assert report["candidate_status"] == "PRODUCTION_CANDIDATE_QUALIFIED"
    assert report["package_status"] == "RELEASE_CANDIDATE_PACKAGE"
    assert report["production_promotion"] is False
    assert report["automatic_retraining"] is False


def test_release_documentation_package_is_complete():
    expected = {
        "system-architecture.md", "model-card.md", "deployment-guide.md",
        "operation-manual.md", "troubleshooting-guide.md", "rollback-procedure.md",
    }
    assert expected <= {path.name for path in (ROOT / "docs/release").glob("*.md")}
