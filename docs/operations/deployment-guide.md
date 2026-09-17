# Deployment Guide

This guide describes a release-candidate environment. It is not production authorization.

## Prerequisites

- Python 3.11 and the qualified dependency lock;
- target-compatible NVIDIA/CUDA runtime for D3 inference;
- separately managed frozen artifacts at manifest-declared paths;
- PostgreSQL for the full backend path;
- Redis for shared metrics, WebSocket fan-out, and persistent Copilot sessions when using multiple workers or replicas;
- approved Camera/PLC/MES simulator or site gateway endpoints.

## Verification sequence

1. Confirm the intended Git revision and a clean working tree.
2. Create an isolated environment and install the exact locked dependencies.
3. Mount model artifacts read-only; do not copy them into Git or the image.
4. Verify release manifest, dependency lock, candidate manifest, evidence files, and artifact hashes.
5. Start PostgreSQL and Redis, set `IVQC_REDIS_URL`, and apply Alembic migrations for the backend path. Staging/production startup fails without Redis.
6. Start inference and require readiness only after a manifest-verified smoke image.
7. Start the decision, review, PLC/MES, and dashboard services.
8. Run a test-product chain and confirm PASS/REJECT/HOLD, traceability, and idempotency.
9. Run the environment-specific local gates before any site release review.

## Local code verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m model_governance.rollback_simulation
docker compose up -d postgres redis
```

Model artifacts and live industrial services are not in the repository; commands that require them are documented local gates rather than fabricated CI passes.

## Abort conditions

Abort startup on missing artifact, hash mismatch, dependency mismatch, smoke failure, invalid frame contract, service readiness failure, or unavailable required field integration. Do not edit hashes, threshold, banks, or manifests to bypass the gate.

## Source procedures

- [D3 release deployment guide](../release/deployment-guide.md)
- [Edge runtime design](../industrial-edge-runtime-design.md)
- [Deployment architecture](../architecture/deployment-architecture.md)
- [Test matrix](../test-matrix.md)
