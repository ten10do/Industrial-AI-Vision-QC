from __future__ import annotations

from pathlib import Path

from app.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DATABASE_URL = "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/vision_qc"


def test_default_database_url_matches_local_compose(monkeypatch):
    monkeypatch.delenv("IVQC_DATABASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url == EXPECTED_DATABASE_URL


def test_example_env_and_compose_match_default_database():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert f"IVQC_DATABASE_URL={EXPECTED_DATABASE_URL}" in env_example
    assert "POSTGRES_DB: vision_qc" in compose
    assert '"127.0.0.1:5433:5432"' in compose


def test_example_env_and_compose_define_shared_redis():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "IVQC_REDIS_URL=redis://127.0.0.1:6380/0" in env_example
    assert "redis:8.2-alpine" in compose
    assert '"127.0.0.1:6380:6379"' in compose
