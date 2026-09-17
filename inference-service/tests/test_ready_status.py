"""Regression: /ready must answer 503, not 200, while not ready.

P0: the readiness probe previously returned HTTP 200 with a body of
{"status": "not_ready", ...}. Load balancers and E2E wait functions that key
off the status code alone would treat an instance that has not loaded its
models as available. The probe must be fail-closed: 503 until the pinned
stack verifies and both models are loadable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def ready_client(monkeypatch):
    from fastapi.testclient import TestClient

    from inference_app import api

    # Never touch real model files: pin both predictors to unloaded so the
    # readiness verdict depends purely on verify_deployment().
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setattr(api, "_anomaly", None)
    return TestClient(api.create_app())


def test_ready_returns_503_when_not_ready(ready_client, monkeypatch):
    import inference_app.api as api_module

    monkeypatch.setattr(
        api_module, "verify_deployment",
        lambda: ["yolo load/smoke failed: weight file not found", "patchcore model not loaded"],
    )
    r = ready_client.get("/ready")
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["problems"], "503 body must carry the problem list for diagnostics"


def test_ready_returns_200_when_ready(ready_client, monkeypatch):
    import inference_app.api as api_module

    monkeypatch.setattr(api_module, "verify_deployment", lambda: [])
    r = ready_client.get("/ready")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["anomaly_loaded"] is True


def test_ready_never_downgrades_a_failed_verification(ready_client, monkeypatch):
    """A 503 must never collapse to 200: the status code is the contract."""
    import inference_app.api as api_module

    monkeypatch.setattr(api_module, "verify_deployment", lambda: ["manifest load failed: boom"])
    r = ready_client.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
