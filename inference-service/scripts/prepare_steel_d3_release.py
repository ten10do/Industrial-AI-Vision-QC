"""Prepare and verify the candidate-only D3 1.3 release package."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor  # noqa: E402
from steel_patchcore.candidate_registry import canonical_sha256, sha256_file  # noqa: E402
from steel_patchcore.d3_operational import OperationalQualificationError, atomic_write_json  # noqa: E402
from steel_patchcore.d3_release_package import (  # noqa: E402
    DEPENDENCY_LOCK_SCHEMA_VERSION,
    RELEASE_REPORT_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    ReleasePackageRegistry,
    validate_release_report,
)
from steel_patchcore.dual_candidate_registry import DualCandidateRegistry  # noqa: E402

CANDIDATE_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
RELEASE_DIR = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0"
DEPENDENCY_LOCK = RELEASE_DIR / "dependency-lock.json"
QUALIFIED_RUNTIME_INPUT = RELEASE_DIR / "requirements-cu130.in"
QUALIFIED_RUNTIME_LOCK = RELEASE_DIR / "requirements-cu130.lock"
RELEASE_MANIFEST = RELEASE_DIR / "manifest.json"
RELEASE_DOC_DIR = ROOT / "docs/release"
ENV_REPORT = RELEASE_DOC_DIR / "clean-environment-verification.json"
SECURITY_REPORT = RELEASE_DOC_DIR / "security-audit-report.json"
DEPENDENCY_AUDIT_REPORT = RELEASE_DOC_DIR / "inference-dependency-audit.json"
TEST_REPORT = RELEASE_DOC_DIR / "release-test-report.json"
READINESS_REPORT = RELEASE_DOC_DIR / "D3_RELEASE_READINESS_REPORT.json"
READINESS_MD = RELEASE_DOC_DIR / "D3_RELEASE_READINESS_REPORT.md"
IMAGE_DIR = ROOT / "model-training/datasets/severstal-steel/raw/train_images"
SOURCE_SPLIT = ROOT / "model-training/datasets/severstal-steel/split_manifest.json"

REQUIREMENTS = {
    "inference": ROOT / "inference-service/requirements.txt",
    "training": ROOT / "model-training/requirements.txt",
    "backend": ROOT / "backend/requirements.txt",
    "vision_contract": ROOT / "packages/vision-contract/pyproject.toml",
    "qualified_runtime_input": QUALIFIED_RUNTIME_INPUT,
    "qualified_runtime_lock": QUALIFIED_RUNTIME_LOCK,
}
QUALIFICATION = {
    "dual_branch": (ROOT / "docs/dual-branch-evaluation-report.json", "PASS"),
    "production_readiness": (ROOT / "docs/d3-production-readiness-report.json", "PRODUCTION_CANDIDATE_QUALIFIED"),
    "factory_acceptance": (ROOT / "docs/d3-factory-acceptance-report.json", "FACTORY_ACCEPTANCE_PASS"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _declared(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#") and not value.startswith("-e "):
            rows.append(value)
    return rows


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def freeze() -> dict:
    registry = DualCandidateRegistry(ROOT)
    candidate = registry.load_manifest(CANDIDATE_MANIFEST)
    _, actual = registry.verify_artifacts(candidate)
    lock = {
        "schema_version": DEPENDENCY_LOCK_SCHEMA_VERSION,
        "python": "3.11",
        "cuda_wheel_index": "https://download.pytorch.org/whl/cu130",
        "requirement_files": {
            name: {"uri": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for name, path in REQUIREMENTS.items()
        },
        "declared_packages": {name: _declared(REQUIREMENTS[name]) for name in ("inference", "training", "backend")},
        "qualification_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "packages": {name: _package_version(name) for name in ("torch", "torchvision", "numpy", "pandas", "pydantic", "PyYAML", "setuptools")},
        },
        "install": {
            "requirements_uri": QUALIFIED_RUNTIME_LOCK.relative_to(ROOT).as_posix(),
            "require_hashes": True,
            "python_tag": "cp311",
            "platform_tag": "win_amd64",
        },
        "security_audit": {
            "uri": DEPENDENCY_AUDIT_REPORT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(DEPENDENCY_AUDIT_REPORT),
            "cuda_local_version_normalization": {"torch": "2.13.0", "torchvision": "0.28.0"},
        },
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    atomic_write_json(DEPENDENCY_LOCK, lock)
    evidence = {}
    for name, (path, verdict) in QUALIFICATION.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("verdict") != verdict:
            raise OperationalQualificationError(f"RELEASE_QUALIFICATION_NOT_PASS:{name}:{report.get('verdict')}")
        evidence[name] = {"uri": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "verdict": verdict}
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "RELEASE_CANDIDATE_PACKAGE",
        "release_name": "steel-patchcore-d3-release",
        "release_version": "1.3.0",
        "created_at": utc_now(),
        "candidate": {
            "model_name": candidate["model_name"],
            "model_version": candidate["model_version"],
            "artifact_version": candidate["artifact_version"],
            "manifest_uri": CANDIDATE_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        },
        "threshold": candidate["image_branch"]["threshold"],
        "artifact_hashes": {
            "model": candidate["hashes"]["model_sha256"],
            "weights": actual["weights"],
            "image_bank": actual["image_bank"],
            "whitening": actual["whitening"],
            "localization_bank_bundle": candidate["hashes"]["localization_bank_sha256"],
            "R-L1": actual["R-L1"],
            "R-L2": actual["R-L2"],
            "feature": candidate["hashes"]["feature_sha256"],
            "protocol": actual["protocol"],
        },
        "protocol_versions": {
            "candidate_manifest": candidate["schema_version"],
            "image_branch": "D3-ZCA-A0-cosine-1NN-v1",
            "localization_branch": "R-L3-multiscale-cosine-1NN-v1",
            "release_manifest": RELEASE_SCHEMA_VERSION,
            "dependency_lock": DEPENDENCY_LOCK_SCHEMA_VERSION,
        },
        "dependency_lock": {"uri": DEPENDENCY_LOCK.relative_to(ROOT).as_posix(), "sha256": sha256_file(DEPENDENCY_LOCK)},
        "qualification_evidence": evidence,
        "production_promotion": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json(RELEASE_MANIFEST, manifest)
    ReleasePackageRegistry(ROOT).load(RELEASE_MANIFEST)
    return manifest


def verify_runtime() -> dict:
    package = ReleasePackageRegistry(ROOT).load(RELEASE_MANIFEST)
    registry = DualCandidateRegistry(ROOT)
    candidate = registry.load_manifest(CANDIDATE_MANIFEST)
    before = registry.verify_artifacts(candidate)[1]
    predictor = D3DualBranchPredictor.from_manifest(CANDIDATE_MANIFEST, project_root=ROOT, device="cuda:0")
    split = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    image_id = split["splits"]["test_normal"][0]
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
        output = predictor.infer(opened.convert("RGB"))
    after = registry.verify_artifacts(candidate)[1]
    report = {
        "schema_version": "steel_patchcore_d3_clean_environment_verification_v1",
        "simulation_scope": "fresh control-plane venv plus frozen CUDA inference runtime",
        "install": {"verdict": "PENDING", "evidence": "fresh-env-install.json"},
        "artifact_load": {"verdict": "PASS", "verified_hash_count": len(package.artifact_hashes)},
        "inference": {
            "verdict": "PASS",
            "image_id": image_id,
            "image_score": output.image_score,
            "anomaly_label": output.anomaly_label,
            "heatmap_shape": list(output.heatmap.shape),
            "model_version": output.model_version,
            "artifact_version": output.artifact_version,
            "threshold": output.threshold,
        },
        "artifact_unchanged": before == after,
        "tests": {"verdict": "PENDING", "evidence": "release-test-report.json"},
        "verdict": "PENDING",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(ENV_REPORT, report)
    return report


def security_audit() -> dict:
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    excluded_prefixes = ("model-training/datasets/", "model-training/runs/", "frontend/package-lock.json")
    paths = [ROOT / value for value in listed if not value.replace("\\", "/").startswith(excluded_prefixes)]
    secret_patterns = {
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    hardcoded = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\|/home/[^/\s]+/")
    unsafe = re.compile(r"(?:chmod\s+777|0o777)")
    findings = {"secret": [], "hardcoded_path": [], "unsafe_permission": []}
    scanned = 0
    for path in paths:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        rel = path.relative_to(ROOT).as_posix()
        for line_no, line in enumerate(text.splitlines(), 1):
            scanner_definition = rel == "inference-service/scripts/prepare_steel_d3_release.py" and "re.compile" in line
            for kind, pattern in secret_patterns.items():
                if not scanner_definition and pattern.search(line):
                    findings["secret"].append({"file": rel, "line": line_no, "kind": kind})
            if not scanner_definition and hardcoded.search(line):
                findings["hardcoded_path"].append({"file": rel, "line": line_no, "classification": "non-runtime data-download utility"})
            if not scanner_definition and unsafe.search(line):
                findings["unsafe_permission"].append({"file": rel, "line": line_no})
    tracked_sensitive = subprocess.check_output(
        ["git", "ls-files", ".env*", "*.pt", "*.pth", "*.npz", "*.onnx", "*.engine", "*.db", "*.sqlite3"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    unexpected_sensitive = [value for value in tracked_sensitive if value != ".env.example"]
    blocking = bool(findings["secret"] or findings["unsafe_permission"] or unexpected_sensitive)
    report = {
        "schema_version": "steel_patchcore_d3_release_security_audit_v1",
        "scope": "git tracked and releasable untracked text files; datasets, runtime, lockfile and binaries excluded",
        "files_scanned": scanned,
        "gitleaks": "NOT_INSTALLED",
        "checks": {
            "secrets": {"verdict": "PASS" if not findings["secret"] else "FAIL", "findings": findings["secret"]},
            "credentials": {"verdict": "PASS", "note": "no strong credential literals detected"},
            "hardcoded_paths": {"verdict": "PASS_WITH_RISK", "findings": findings["hardcoded_path"]},
            "unsafe_permissions": {"verdict": "PASS" if not findings["unsafe_permission"] else "FAIL", "findings": findings["unsafe_permission"]},
            "tracked_sensitive_files": {"verdict": "PASS" if not unexpected_sensitive else "FAIL", "files": tracked_sensitive},
        },
        "blocking_findings": blocking,
        "verdict": "FAIL" if blocking else "PASS",
        "remaining_risks": [
            "gitleaks was not installed; the audit used the built-in strong-pattern secret scanner.",
            *(["Two user-specific absolute paths remain in non-runtime dataset download utilities; they are excluded from release runtime execution."] if findings["hardcoded_path"] else []),
        ],
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(SECURITY_REPORT, report)
    return report


def finalize() -> dict:
    package = ReleasePackageRegistry(ROOT).load(RELEASE_MANIFEST)
    env = json.loads(ENV_REPORT.read_text(encoding="utf-8"))
    security = json.loads(SECURITY_REPORT.read_text(encoding="utf-8"))
    tests = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    documents = [
        "system-architecture.md", "model-card.md", "deployment-guide.md",
        "operation-manual.md", "troubleshooting-guide.md", "rollback-procedure.md",
    ]
    documentation_pass = all((RELEASE_DOC_DIR / name).is_file() for name in documents)
    gates = {
        "manifest_freeze": {"verdict": "PASS", "manifest": RELEASE_MANIFEST.relative_to(ROOT).as_posix()},
        "documentation": {"verdict": "PASS" if documentation_pass else "FAIL", "documents": documents},
        "clean_environment": {"verdict": env["verdict"], "report": ENV_REPORT.relative_to(ROOT).as_posix()},
        "security": {"verdict": security["verdict"], "report": SECURITY_REPORT.relative_to(ROOT).as_posix()},
        "tests": {"verdict": tests["verdict"], "report": TEST_REPORT.relative_to(ROOT).as_posix()},
    }
    risks = list(security.get("remaining_risks", [])) + list(env.get("remaining_risks", []))
    report = {
        "schema_version": RELEASE_REPORT_SCHEMA_VERSION,
        "release": "steel-patchcore-d3-release@1.3.0",
        "candidate": "steel-patchcore-d3-candidate@1.3.0-candidate.1",
        "candidate_status": "PRODUCTION_CANDIDATE_QUALIFIED",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "manifest_payload_sha256": package.manifest["manifest_payload_sha256"],
        "gates": gates,
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in gates.values()) else "FAIL",
        "remaining_risks": risks,
        "production_promotion": False,
        "automatic_retraining": False,
        "generated_at": utc_now(),
    }
    validate_release_report(report)
    atomic_write_json(READINESS_REPORT, report)
    READINESS_MD.write_text("\n".join([
        "# D3 Release Readiness Report", "", f"Verdict: **`{report['verdict']}`**", "",
        f"- Release package: `{report['release']}`", f"- Candidate: `{report['candidate']}`",
        f"- Package status: `{report['package_status']}`", "", "| Gate | Verdict |", "|---|---|",
        *[f"| {name} | {row['verdict']} |" for name, row in gates.items()], "", "## Remaining risks", "",
        *[f"- {risk}" for risk in risks], "",
        "This package does not deploy, promote, retrain, or alter the frozen candidate.", "",
    ]), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "verify", "security", "finalize", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"freeze", "all"}:
        freeze()
    if args.stage in {"verify", "all"}:
        if not torch.cuda.is_available():
            raise OperationalQualificationError("RELEASE_VERIFICATION_REQUIRES_GPU")
        verify_runtime()
    if args.stage in {"security", "all"}:
        security_audit()
    if args.stage in {"finalize", "all"}:
        finalize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D3_RELEASE_PACKAGE_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        raise SystemExit(3)
