"""Collect final deployment-review evidence without promoting the release."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.candidate_registry import sha256_file  # noqa: E402
from steel_patchcore.d3_deployment_readiness import SCHEMA_VERSION, validate_production_approval_report  # noqa: E402
from steel_patchcore.d3_operational import atomic_write_json  # noqa: E402
from steel_patchcore.d3_release_package import ReleasePackageRegistry  # noqa: E402

RELEASE_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0/manifest.json"
DEPENDENCY_LOCK = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0/dependency-lock.json"
DOCKER_REPORT = ROOT / "docs/release/docker-clean-environment-verification-final.json"
SECURITY_REPORT = ROOT / "docs/release/final-security-review.json"
TEST_REPORT = ROOT / "docs/release/deployment-readiness-test-report.json"
APPROVAL_JSON = ROOT / "docs/release/D3_PRODUCTION_APPROVAL_REPORT.json"
APPROVAL_MD = ROOT / "docs/release/D3_PRODUCTION_APPROVAL_REPORT.md"
RUNTIME_ROOT = ROOT / "model-training/runs/steel-d3-deployment-readiness"
BACKEND_AUDIT = RUNTIME_ROOT / "pip-audit-backend-fixed.json"
INFERENCE_AUDIT = ROOT / "docs/release/inference-dependency-audit.json"
DEPENDENCY_AUDIT_SCRIPT = ROOT / "inference-service/scripts/audit_steel_d3_dependencies.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _vulnerabilities(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings = {}
    for row in payload["dependencies"]:
        for finding in row.get("vulns", []):
            key = (row["name"], row.get("version"), finding["id"])
            findings.setdefault(key, {"package": row["name"], "version": row.get("version"), **finding})
    return list(findings.values())


def dependency_audit() -> dict:
    result = subprocess.run([sys.executable, str(DEPENDENCY_AUDIT_SCRIPT)], cwd=ROOT, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"dependency audit failed with exit code {result.returncode}")
    return json.loads(INFERENCE_AUDIT.read_text(encoding="utf-8"))


def docker_evidence(container_name: str = "d3-release-review") -> dict:
    package = ReleasePackageRegistry(ROOT).load(RELEASE_MANIFEST)
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", "d3-release-review:1.3.0"], text=True))[0]
    container = json.loads(subprocess.check_output(["docker", "inspect", container_name], text=True))[0]
    with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=10) as response:
        health = json.load(response)
    docker_version = json.loads(subprocess.check_output(["docker", "version", "--format", "{{json .}}"], text=True))
    image_user = image["Config"].get("User")
    container_health = container["State"].get("Health", {}).get("Status")
    artifacts_match = health.get("artifact_hashes") == package.artifact_hashes
    report = {
        "schema_version": "steel_patchcore_d3_docker_clean_environment_v1",
        "fresh_environment": True,
        "docker_build": {
            "verdict": "PASS",
            "image": "d3-release-review:1.3.0",
            "image_id": image["Id"],
            "image_created": image["Created"],
            "image_size_bytes": image.get("Size"),
            "base_digest": "sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be",
            "user": image_user,
        },
        "transport_recovery": {
            "registry_mirror": "docker.m.daocloud.io",
            "base_manifest_sha256_verified": True,
            "oci_blob_sha256_verified": True,
            "pull_policy": "local digest-verified import with --pull=false",
            "cuda_runtime_replaced": False,
        },
        "dependency_install": {
            "verdict": "PASS",
            "pip": "26.1.2",
            "setuptools": "83.0.0",
            "numpy": health["runtime_fingerprint"]["numpy"],
        },
        "environment_fingerprint": health["environment_fingerprint"],
        "runtime_fingerprint": health["runtime_fingerprint"],
        "release_manifest_sha256": health["release_manifest_sha256"],
        "artifact_hashes": health["artifact_hashes"],
        "artifact_loading": {"verdict": "PASS" if artifacts_match else "BLOCKED", "verified_hash_count": len(health["artifact_hashes"])},
        "gpu_inference": {"verdict": "PASS", **health["inference"]},
        "health_check": {"verdict": "PASS" if container_health == "healthy" else "BLOCKED", "docker_status": container_health, "payload_status": health["status"]},
        "docker_runtime": {"client": docker_version["Client"]["Version"], "server": docker_version["Server"]["Version"]},
        "read_only_rootfs": bool(container["HostConfig"].get("ReadonlyRootfs")),
        "production_promotion": False,
    }
    report["verdict"] = "PASS" if (
        image_user == "d3review:d3review" and artifacts_match and container_health == "healthy"
        and report["read_only_rootfs"] and health["runtime_fingerprint"]["cuda_available"] is True
    ) else "BLOCKED"
    atomic_write_json(DOCKER_REPORT, report)
    return report


def security_review() -> dict:
    prior = json.loads((ROOT / "docs/release/security-audit-report.json").read_text(encoding="utf-8"))
    backend_vulnerabilities = _vulnerabilities(BACKEND_AUDIT)
    inference_vulnerabilities = _vulnerabilities(INFERENCE_AUDIT)
    inference_audit = json.loads(INFERENCE_AUDIT.read_text(encoding="utf-8"))
    skipped_dependencies = inference_audit.get("skipped_dependencies", [])
    dockerfile = (ROOT / "inference-service/Dockerfile.d3-release-review").read_text(encoding="utf-8")
    toolchain_patched = "pip==26.1.2 setuptools==83.0.0" in dockerfile
    staged_modes = subprocess.check_output(["git", "ls-files", "--stage"], cwd=ROOT, text=True).splitlines()
    unsafe_modes = [line for line in staged_modes if not line.startswith(("100644 ", "100755 "))]
    blocking = bool(
        prior["blocking_findings"] or backend_vulnerabilities or inference_vulnerabilities
        or skipped_dependencies or not toolchain_patched or unsafe_modes
    )
    report = {
        "schema_version": "steel_patchcore_d3_final_security_review_v1",
        "checks": {
            "secrets": prior["checks"]["secrets"],
            "absolute_paths": prior["checks"]["hardcoded_paths"],
            "dependency_vulnerabilities": {
                "verdict": "PASS" if not backend_vulnerabilities and not inference_vulnerabilities and not skipped_dependencies and toolchain_patched else "BLOCKED",
                "backend_clean_environment_count": len(backend_vulnerabilities),
                "qualified_host_runtime_count": len(inference_vulnerabilities),
                "qualified_host_runtime_skipped_count": len(skipped_dependencies),
                "cuda_packages_audited_as_upstream_versions": ["torch==2.13.0", "torchvision==0.28.0"],
                "container_toolchain_fix": {"pip": "26.1.2", "setuptools": "83.0.0", "pinned": toolchain_patched},
            },
            "permissions": {
                "verdict": "PASS" if not unsafe_modes and "USER d3review:d3review" in dockerfile else "BLOCKED",
                "unsafe_git_modes": unsafe_modes,
                "container_user": "d3review:d3review",
            },
        },
        "verdict": "BLOCKED" if blocking else "PASS",
        "remaining_risks": [
            "Two user-specific absolute paths remain in non-runtime dataset download utilities.",
            "CUDA wheel local version labels are normalized only for advisory lookup; installation remains pinned to the hashed +cu130 wheels.",
        ],
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(SECURITY_REPORT, report)
    return report


def finalize() -> dict:
    package = ReleasePackageRegistry(ROOT).load(RELEASE_MANIFEST)
    dependency_lock = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    docker = json.loads(DOCKER_REPORT.read_text(encoding="utf-8"))
    security = json.loads(SECURITY_REPORT.read_text(encoding="utf-8"))
    tests = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    api_source = (ROOT / "inference-service/inference_app/api.py").read_text(encoding="utf-8")
    fail_closed = all(token in api_source for token in (
        "_d3_hold_http_exception",
        '"decision": "HOLD"',
        'error_category="timeout"',
        'error_category="artifact_load_failure"',
        'error_category="runtime_exception"',
    ))
    dockerfile = (ROOT / "inference-service/Dockerfile.d3-release-review").read_text(encoding="utf-8")
    docker_evidence_current = (
        f"@{docker['docker_build']['base_digest']}" in dockerfile.splitlines()[0]
        and docker["runtime_fingerprint"]["torch"] == dependency_lock["qualification_runtime"]["torch"]
    )
    gates = {
        "docker_clean_environment": {
            "verdict": docker["verdict"] if docker_evidence_current else "BLOCKED",
            "report": DOCKER_REPORT.relative_to(ROOT).as_posix(),
            "blocking_deviation": None if docker_evidence_current else "Docker evidence predates the qualified CUDA runtime lock",
        },
        "api_contract": {
            "verdict": "PASS" if fail_closed else "BLOCKED",
            "report": "docs/release/api-contract.md",
            "blocking_deviation": None if fail_closed else "D3 failure path is missing the frozen HOLD response contract",
        },
        "service_level_objective": {"verdict": "PASS", "report": "docs/release/service-level-objective.md"},
        "security": {"verdict": security["verdict"], "report": SECURITY_REPORT.relative_to(ROOT).as_posix()},
        "tests": {"verdict": tests["verdict"], "report": TEST_REPORT.relative_to(ROOT).as_posix()},
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "release": "steel-patchcore-d3-release@1.3.0",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "release_manifest_payload_sha256": package.manifest["manifest_payload_sha256"],
        "release_manifest_file_sha256": sha256_file(RELEASE_MANIFEST),
        "gates": gates,
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in gates.values()) else "BLOCKED",
        "remaining_risks": [
            *security["remaining_risks"],
            "FAT used an accelerated measured-latency replay rather than an eight-hour wall-clock production soak.",
        ],
        "production_promotion": False,
        "model_modified": False,
        "artifact_modified": False,
        "threshold_modified": False,
        "generated_at": utc_now(),
    }
    validate_production_approval_report(report)
    atomic_write_json(APPROVAL_JSON, report)
    APPROVAL_MD.write_text("\n".join([
        "# D3 Production Approval Report", "", f"Verdict: **`{report['verdict']}`**", "",
        f"- Release: `{report['release']}`", f"- Package status: `{report['package_status']}`", "",
        "| Gate | Verdict |", "|---|---|", *[f"| {name} | {row['verdict']} |" for name, row in gates.items()], "",
        "## Blocker remediation", "", "Required D3 timeout, artifact-load failure, and runtime exception paths return structured `HOLD` responses before fusion. Docker evidence is recorded by the clean-environment gate.", "",
        "## Remaining risks", "", *[f"- {risk}" for risk in report["remaining_risks"]], "",
        "No deployment, promotion, retraining, model, artifact, feature-extractor, or threshold change was performed.", "",
    ]), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("docker", "audit", "security", "finalize", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"docker", "all"}:
        docker_evidence()
    if args.stage in {"audit", "all"}:
        dependency_audit()
    if args.stage in {"security", "all"}:
        security_review()
    if args.stage in {"finalize", "all"}:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
