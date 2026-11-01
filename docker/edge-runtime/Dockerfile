# Industrial edge runtime container (simulation of an industrial PC deployment).
#
# Deliberately a slim CPU image: the D3 artifacts are NEVER baked into the
# image - they are mounted read-only at runtime, so the frozen candidate
# manifest/weights/bank stay outside the container lifecycle:
#
#   docker run \
#     -v /opt/ivqc/models:/models:ro \
#     -e IVQC_D3_CANDIDATE_MANIFEST=/models/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json \
#     -e INDUSTRIAL_EDGE_CONFIG=/app/industrial_runtime/edge_config.yaml \
#     -p 8080:8080 \
#     ivqc-edge-runtime:1.0.0
#
FROM python:3.15-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDUSTRIAL_EDGE_CONFIG=/app/industrial_runtime/edge_config.yaml

WORKDIR /app

# Runtime dependencies only (no CUDA base needed by the runtime layer itself;
# GPU inference stays in the unchanged inference service deployment).
RUN pip install --no-cache-dir \
    pydantic==2.13.4 \
    fastapi==0.141.1 \
    uvicorn==0.52.1 \
    httpx==0.28.1 \
    PyYAML==6.0.3 \
    psutil==7.2.2

# Edge runtime source + drift monitoring package + healthcheck probe.
COPY industrial_runtime/ /app/industrial_runtime/
COPY monitoring/ /app/monitoring/
COPY industrial_loop/ /app/industrial_loop/
COPY docker/edge-runtime/healthcheck.py /app/healthcheck.py

# Model/artifact mount point (read-only in production).
VOLUME ["/models"]

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

CMD ["python", "-m", "industrial_runtime.service", "--host", "0.0.0.0", "--port", "8080"]
