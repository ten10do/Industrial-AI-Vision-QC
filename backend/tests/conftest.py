from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Phase 7: unit tests run with the PLC/MES integration disabled (the
# industrial closed loop is covered by dedicated integration tests with the
# real simulators on 8501/8502).
os.environ.setdefault("IVQC_PLC_ENABLED", "false")
os.environ.setdefault("IVQC_MES_ENABLED", "false")

# Governance credentials for the test suite. Configured here, before app
# import, because the settings object is cached on first use. Without them the
# model registry is closed, which is the production default.
os.environ.setdefault("IVQC_PIPELINE_HMAC_SECRET", "test-pipeline-secret")
os.environ.setdefault("IVQC_RUNTIME_ENV_FILE", "backend/tests/.env.runtime.test")
os.environ.setdefault(
    "IVQC_API_TOKENS",
    json.dumps(
        {
            "test-viewer-token": {"subject": "tester-viewer", "roles": ["viewer"]},
            "test-engineer-token": {"subject": "tester-engineer", "roles": ["engineer"]},
            "test-pipeline-token": {"subject": "tester-pipeline", "roles": ["pipeline"]},
            "test-approver-token": {"subject": "tester-approver", "roles": ["approver"]},
            "test-admin-token": {"subject": "tester-admin", "roles": ["admin"]},
            "test-operator-token": {"subject": "tester-operator", "roles": ["operator"]},
            "test-reviewer-a-token": {"subject": "tester-reviewer-a", "roles": ["reviewer"]},
            "test-reviewer-b-token": {"subject": "tester-reviewer-b", "roles": ["reviewer"]},
            "test-release-manager-token": {"subject": "tester-release-manager", "roles": ["release-manager"]},
        }
    ),
)

from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_session  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def app():
    application = create_app()
    yield application
    application.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app, db_session):
    """Authenticated client. Carries the operator bearer token by default so
    shop-floor business tests run as a logged-in operator; tests that need a
    different role (reviewer / pipeline / release-manager / admin) pass
    explicit headers. Use `client_unauthenticated` (or drop the header) for
    fail-closed assertions."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TEST_TOKENS['operator']}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers) as ac:
        yield ac


@pytest_asyncio.fixture
async def client_unauthenticated(app, db_session):
    """No Authorization header: every protected endpoint must answer 401."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


TEST_TOKENS = {
    "viewer": "test-viewer-token",
    "engineer": "test-engineer-token",
    "pipeline": "test-pipeline-token",
    "approver": "test-approver-token",
    "admin": "test-admin-token",
    "operator": "test-operator-token",
    "reviewer_a": "test-reviewer-a-token",
    "reviewer_b": "test-reviewer-b-token",
    "release-manager": "test-release-manager-token",
}

ARTIFACT_URI = "inference-service/models/best.pt"

# Deterministic stand-in content for a fresh checkout. The real model file is
# gitignored (datasets and model artifacts must never enter git), so CI does
# not have it; the governance tests never load the model, they only exercise
# server-side re-hashing, which needs *a* file at the registered path whose
# bytes hash to the registered digest.
_ARTIFACT_STANDIN = b"ivqc synthetic test artifact\n" * 64


@pytest.fixture
def auth():
    """Bearer headers for a role: auth("approver") -> {"Authorization": ...}."""

    def _headers(role: str = "admin") -> dict:
        return {"Authorization": f"Bearer {TEST_TOKENS[role]}"}

    return _headers


@pytest.fixture(scope="session")
def artifact():
    """An artifact plus the SHA256 the server will recompute for it.

    The file must exist on every machine that runs the suite, and the real
    model does not: it is gitignored, so a fresh checkout (CI) has no
    ``best.pt`` at all, which is why backend-ci failed while every developer
    machine passed. When the real artifact is absent this fixture writes a
    deterministic stand-in at the registered path and removes it at session
    teardown; it never overwrites a real artifact. Tests that genuinely need
    the deployed bytes carry the ``artifact`` marker instead, which CI
    excludes explicitly (see test_manifest_artifact_sha256_matches_files)."""
    import hashlib

    path = Path(__file__).resolve().parents[2] / ARTIFACT_URI
    created = False
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_ARTIFACT_STANDIN)
        created = True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    try:
        yield {"uri": ARTIFACT_URI, "sha256": h.hexdigest()}
    finally:
        if created:
            # Cleanup must never fail a test (sandboxed unlink can raise);
            # the path is gitignored, so a leak is contained.
            try:
                path.unlink()
            except OSError:
                pass


@pytest.fixture
def eval_report():
    """An eval report the domain-validation claim can point at. Lives under
    the project root so the server can resolve and re-hash it relative to that
    root, which is why a tmp_path file would not do."""
    import hashlib
    import json as _json

    target = Path(__file__).resolve().parents[1] / ".artifacts"
    target.mkdir(exist_ok=True)
    file = target / f"eval-report-{uuid4().hex[:10]}.json"
    file.write_text(_json.dumps({"domain": "steel", "sample_count": 42}), encoding="utf-8")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    rel = file.relative_to(Path(__file__).resolve().parents[2]).as_posix()
    try:
        yield {"uri": rel, "sha256": digest}
    finally:
        # Cleanup must never turn a passing test into an ERROR. Some sandboxed
        # environments intercept unlink() and fail the trash operation with
        # OSError (observed: SHFileOperationW 0x2 on Windows). The file is
        # disposable and backend/.artifacts/ is gitignored, so a leak here is
        # contained and strictly preferable to masking a real test result.
        try:
            file.unlink(missing_ok=True)
        except OSError:
            pass


@pytest.fixture
def stub_infer(app):
    """Install a stub inference client through dependency overrides."""

    def _install(stub):
        from app.api.inspections import get_inspection_service
        from app.services.inspection_service import InspectionService

        app.dependency_overrides[get_inspection_service] = lambda: InspectionService(inference_client=stub)
        return stub

    return _install
