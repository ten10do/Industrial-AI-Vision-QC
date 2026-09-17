from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_oversized_upload_returns_413_before_model_load(monkeypatch):
    from inference_app import api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)

    def unexpected_model_load():
        raise AssertionError("model must not load for an oversized upload")

    monkeypatch.setattr(api, "get_predictor", unexpected_model_load)
    response = TestClient(api.create_app()).post(
        "/v1/infer",
        files={"file": ("too-large.jpg", b"12345", "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["error"]["code"] == "payload_too_large"
