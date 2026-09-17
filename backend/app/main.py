from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .api import inspections, products, quality_rules, realtime, reviews
from .config import get_settings
from .database import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="IndustrialVision-QC Backend", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "app": settings.app_name}

    @app.get("/ready")
    async def ready() -> dict:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "ready", "database": "ok"}
        except Exception:
            logger.exception("ready check failed")
            return JSONResponse(status_code=503, content={"status": "not_ready", "database": "unreachable"})

    app.include_router(inspections.router)
    app.include_router(products.router)
    app.include_router(quality_rules.router)
    app.include_router(realtime.rt_router)
    app.include_router(realtime.ws_router)
    app.include_router(reviews.router)
    from .api import models_registry

    app.include_router(models_registry.router)
    from .api import evaluations

    app.include_router(evaluations.router)
    from .api import mlops_monitoring

    app.include_router(mlops_monitoring.router)
    from .api import copilot

    app.include_router(copilot.router)

    @app.exception_handler(FastAPIHTTPException)
    async def http_error_handler(request: Request, exc: FastAPIHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            body = exc.detail
        elif isinstance(exc.detail, dict):
            body = {"error": {**exc.detail}}
        else:
            body = {"error": {"code": "http_error", "message": str(exc.detail)}}
        if "request_id" not in body["error"]:
            body["error"]["request_id"] = _rid(request)
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "unexpected server error", "request_id": _rid(request)}},
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time

        start = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %d %.1fms", request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
        return response

    return app


def _rid(request: Request) -> str:
    return request.headers.get("X-Request-ID", "unknown")


app = create_app()
