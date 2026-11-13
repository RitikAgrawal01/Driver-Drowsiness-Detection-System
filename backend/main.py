"""
backend/main.py — FastAPI app: WebSocket monitoring, REST predict, health, metrics, read-only model info.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.config import get_settings
from backend.logger import get_logger, setup_logging
from backend.routers import models, predict, session, stream
from backend.services import model_client
from backend.services.serving import serving
from backend.services.session_manager import get_session_manager

settings = get_settings()
setup_logging(settings.log_level, settings.environment)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("backend starting", version=settings.version, environment=settings.environment)
    await serving.refresh()

    async def refresher():  # picks up model hot-reloads and model-server recovery
        while True:
            await asyncio.sleep(30)
            try:
                await serving.refresh()
            except Exception as e:
                logger.error("serving refresh failed", error=str(e))

    task = asyncio.create_task(refresher())
    yield
    task.cancel()
    get_session_manager().close_all()
    await model_client.close_client()


app = FastAPI(title="Driver Drowsiness Detection — Backend API", version=settings.version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(stream.router)
app.include_router(predict.router)
app.include_router(session.router)
app.include_router(models.router)


@app.get("/health", tags=["Health"])
async def health():
    """Liveness: the process is up."""
    return {"status": "ok", "service": "backend", "version": settings.version}


@app.get("/ready", tags=["Health"])
async def ready():
    """Readiness: the model server is serving a model whose settings match ours."""
    await serving.refresh()
    if not serving.ready:
        raise HTTPException(status_code=503, detail=serving.config_error or "model server not ready")
    return {"status": "ready", "model_version": serving.model["model_version"], "window_s": serving.model["window_s"]}


@app.get("/status", tags=["Health"])
async def status():
    return {
        "status": "ok",
        "version": settings.version,
        "active_sessions": get_session_manager().active_count(),
        "model": serving.model,
        "config_error": serving.config_error,
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
