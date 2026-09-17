"""Final D3 deployment-readiness package contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from vision_contract import InferenceResult, utc_now_iso

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from inference_app import api  # noqa: E402
from steel_patchcore.d3_deployment_readiness import SCHEMA_VERSION, validate_production_approval_report  # noqa: E402


class _YoloStub:
    device = "cpu"

    @staticmethod
    def _read_image(data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def predict(self, image: np.ndarray, inspection_id: str | None = None) -> InferenceResult:
        return InferenceResult(
            inspection_id=inspection_id or "insp-d3-fail-closed",
            model_name="yolov8s",
            model_version="phase1-baseline",
            image_width=image.shape[1],
            image_height=image.shape[0],
            detections=[],
            inference_latency_ms=0.1,
            device="cpu",
            timestamp=utc_now_iso(),
        )


class _AnomalyFailureStub:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def predict(self, *args, **kwargs):
        raise self.error


def _post_infer(
    monkeypatch,
    anomaly_loader=None,
    trace_id: str = "trace-d3-blocker",
    candidate_manifest: str = "configured-d3-candidate.json",
):
    monkeypatch.setattr(api, "D3_CANDIDATE_MANIFEST", candidate_manifest)
    monkeypatch.setattr(api, "_anomaly", None)
    monkeypatch.setattr(api, "get_predictor", lambda: _YoloStub())
    if anomaly_loader is not None:
        monkeypatch.setattr(api, "get_anomaly_predictor", anomaly_loader)
    monkeypatch.setattr(api, "fuse", lambda *args, **kwargs: pytest.fail("fusion must not run after D3 failure"))
    ok, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return TestClient(api.app).post(
        "/v1/infer",
        files={"file": ("sample.png", encoded.tobytes(), "image/png")},
        headers={"X-Request-ID": trace_id},
    )


def _assert_hold(response, *, category: str, reason_code: str, trace_id: str = "trace-d3-blocker"):
    assert response.status_code == 503
    error = response.json()["detail"]["error"]
    assert error == {
        "code": "d3_inference_failed",
        "message": error["message"],
        "request_id": trace_id,
        "trace_id": trace_id,
        "reason_code": reason_code,
        "d3_status": "FAILED",
        "error_category": category,
        "decision": "HOLD",
    }


def test_docker_review_image_is_pinned_non_root_and_health_checked():
    dockerfile = (ROOT / "inference-service/Dockerfile.d3-release-review").read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:" in dockerfile
    assert "USER d3review:d3review" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY model-training/runs" not in dockerfile


def test_api_contract_freezes_input_output_errors_trace_and_decision():
    text = (ROOT / "docs/release/api-contract.md").read_text(encoding="utf-8")
    for required in (
        "POST /v1/infer", "multipart/form-data", "VisionResult", "invalid_image",
        "model_unavailable", "vision_error", "d3_inference_failed", "X-Request-ID",
        "reason_code", "trace_id", "d3_status", "error_category", "REVIEW_REQUIRED", "HOLD",
    ):
        assert required in text


def test_slo_defines_latency_availability_rollback_and_monitoring():
    text = (ROOT / "docs/release/service-level-objective.md").read_text(encoding="utf-8")
    for required in ("99.5%", "p95", "750 ms", "2000 ms", "15 minutes", "Artifact integrity", "Drift"):
        assert required in text


def test_production_approval_verdict_is_derived_and_never_promotes():
    gates = {name: {"verdict": "PASS"} for name in (
        "docker_clean_environment", "api_contract", "service_level_objective", "security", "tests"
    )}
    report = {
        "schema_version": SCHEMA_VERSION,
        "release": "steel-patchcore-d3-release@1.3.0",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "gates": gates,
        "verdict": "PASS",
        "remaining_risks": [],
        "production_promotion": False,
        "model_modified": False,
    }
    validate_production_approval_report(report)
    report["gates"]["security"]["verdict"] = "BLOCKED"
    report["verdict"] = "BLOCKED"
    validate_production_approval_report(report)


def test_d3_timeout_fails_closed_to_hold(monkeypatch):
    response = _post_infer(monkeypatch, lambda: _AnomalyFailureStub(TimeoutError("deadline exceeded")))
    _assert_hold(response, category="timeout", reason_code="d3_inference_timeout")


def test_d3_artifact_load_failure_fails_closed_to_hold(monkeypatch, tmp_path):
    response = _post_infer(
        monkeypatch,
        candidate_manifest=str(tmp_path / "missing-candidate-manifest.json"),
    )
    _assert_hold(response, category="artifact_load_failure", reason_code="d3_artifact_load_failure")


def test_d3_runtime_exception_fails_closed_to_hold(monkeypatch):
    response = _post_infer(monkeypatch, lambda: _AnomalyFailureStub(RuntimeError("CUDA kernel failure")))
    _assert_hold(response, category="runtime_exception", reason_code="d3_runtime_failure")


def test_d3_failure_path_is_fail_closed_in_source():
    source = (ROOT / "inference-service/inference_app/api.py").read_text(encoding="utf-8")
    assert "_d3_hold_http_exception" in source
    assert '"decision": "HOLD"' in source
    assert "the anomaly channel is best-effort" not in source
