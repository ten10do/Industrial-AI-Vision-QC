# D3 Release Deployment Guide

This guide prepares a release candidate environment. It does not authorize production deployment or registry promotion.

## Prerequisites

- Python 3.11 on the qualified Windows/CUDA platform.
- NVIDIA driver compatible with CUDA 12.8 and an RTX 5060-class validated GPU.
- Artifact files at the URIs recorded in the candidate manifest.
- No model weights, banks or datasets copied into Git.

## Verify the package

1. Create an isolated Python environment.
2. Install the qualified inference runtime from the hash lock referenced by `dependency-lock.json`: `python -m pip install --require-hashes -r model-training/registry/steel-patchcore-d3-release/1.3.0/requirements-cu130.lock`. The lock includes the CUDA 13.0 index and must be used on CPython 3.11 / Windows x86-64. On Windows, create the runtime in a short path such as `D:\ivqc-d3-runtime`; the Torch wheel contains deeply nested license paths that can exceed the legacy path limit when the virtual environment is under a long checkout path. Install backend requirements separately while the working directory is `backend/`, because its editable `../packages/vision-contract` reference is relative to that directory.
3. Install `packages/vision-contract` from the frozen source tree.
4. Run the release package loader. It must verify the dependency lock, candidate manifest, qualification evidence and every artifact hash before model construction.
5. Run one 1600×256 smoke image and confirm the output contains `image_score`, `anomaly_label`, `heatmap`, `confidence`, `model_version` and `artifact_version`.
6. Run the steel, inference and backend suites.

Before release review, regenerate `docs/release/inference-dependency-audit.json` with `python inference-service/scripts/audit_steel_d3_dependencies.py`. The audit maps the CUDA-local versions (`torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`) to their upstream versions only for advisory lookup; the install lock remains fixed to the hashed CUDA wheels. Any skipped dependency or known vulnerability fails the audit.

## Candidate-only start sequence

```text
dependency lock verification
  -> release manifest verification
  -> candidate/artifact hash verification
  -> model load
  -> sealed smoke inference
  -> READY_FOR_MANUAL_RELEASE_REVIEW
```

Any mismatch must stop the sequence. Do not repair a mismatch by editing a hash, threshold or artifact. Restore the exact approved package or execute the rollback procedure.

## Configuration boundaries

- Keep the candidate manifest path explicit; do not point the production registry at this release automatically.
- Do not enable an automatic promotion job.
- Do not expose writable model, bank or whitening paths to the inference service.
- Persist prediction and monitoring logs outside the source tree.
