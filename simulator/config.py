from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SimulatorConfig:
    source_directory: str = os.environ.get("IVQC_SRC_DIR", "model-training/datasets/neu-det-yolo/test/images")
    interval_ms: int = int(os.environ.get("IVQC_INTERVAL_MS", "500"))
    production_line: str = os.environ.get("IVQC_LINE", "line-a")
    station: str = os.environ.get("IVQC_STATION", "qc-01")
    batch_id: str = os.environ.get("IVQC_BATCH", "batch-p3-001")
    loop: bool = os.environ.get("IVQC_LOOP", "true").lower() in ("1", "true", "yes")
    random_seed: int = int(os.environ.get("IVQC_SEED", "42"))
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    shuffle: bool = True


@dataclass
class OrchestratorConfig:
    backend_url: str = os.environ.get("IVQC_BACKEND_URL", "http://127.0.0.1:8123")
    # Bearer credentials for the fail-closed backend (P0 security). Two
    # distinct identities are required because a single bearer token carries
    # one role set: an operator token for inspection creation and a pipeline
    # token (internal service identity) for telemetry ingestion.
    api_token: str = os.environ.get("IVQC_API_TOKEN", "")
    pipeline_token: str = os.environ.get("IVQC_PIPELINE_TOKEN", "")
    queue_size: int = int(os.environ.get("IVQC_QUEUE_SIZE", "20"))
    workers: int = int(os.environ.get("IVQC_WORKERS", "2"))
    retry_max: int = int(os.environ.get("IVQC_RETRY_MAX", "2"))
    retry_base_ms: float = float(os.environ.get("IVQC_RETRY_BASE_MS", "300"))
    request_timeout_seconds: float = float(os.environ.get("IVQC_TIMEOUT", "60"))
    queue_full_policy: str = "block"  # v1: block the producer, never drop
    telemetry_interval_seconds: float = 2.0
    max_images: int | None = None  # None = run until simulator source exhausted (loop off) or stopped
