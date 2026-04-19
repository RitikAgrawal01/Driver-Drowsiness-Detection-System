"""
backend/main.py
────────────────
FastAPI main application entry point.
Registers all routers, middleware, lifespan events.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from backend.config import get_settings
from backend.logger import setup_logging, get_logger
from backend.metrics import MODEL_SERVER_REACHABLE
from backend.routers import session, predict, stream
from backend.services import model_client
from backend.services.session_manager import get_session_manager

settings = get_settings()
setup_logging(settings.log_level, settings.environment)
logger = get_logger("main")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info(
        f"Starting {settings.project_name} backend",
        version=settings.version,
        environment=settings.environment,
    )

    # Initialise session manager
    get_session_manager()

    # Probe model server health
    reachable = await model_client.health_check()
    if reachable:
        logger.info("Model server reachable ✓")
        MODEL_SERVER_REACHABLE.set(1)
    else:
        logger.warning(
            "Model server NOT reachable at startup — "
            "will retry on each request"
        )

    # Background task: periodic model server health probe every 30s
    async def probe_model_server():
        while True:
            await asyncio.sleep(30)
            await model_client.health_check()

    probe_task = asyncio.create_task(probe_model_server())

    yield  # ← application runs here

    # Shutdown
    logger.info("Backend shutting down...")
    probe_task.cancel()
    get_session_manager().cleanup_all()
    await model_client.close_client()
    logger.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Driver Drowsiness Detection — Backend API",
    description=(
        "Main backend API. Receives webcam frames via WebSocket, "
        "extracts drowsiness features with MediaPipe, calls the model server, "
        "and manages monitoring sessions."
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(session.router)
app.include_router(predict.router)
app.include_router(stream.router)


# ── Health endpoints ──────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    """Docker healthcheck — always returns 200."""
    return {
        "status": "ok",
        "service": "backend",
        "version": settings.version,
    }


@app.get("/ready", tags=["Health"])
async def ready():
    """Readiness — ok only when model server is reachable."""
    reachable = await model_client.health_check()
    if not reachable:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Model server unavailable",
        )
    return {
        "status": "ready",
        "model_server_reachable": True,
        "model_server_url": settings.model_server_url,
    }


@app.get("/status", tags=["Health"])
async def status():
    """Overall system status including active sessions."""
    manager = get_session_manager()
    return {
        "status": "ok",
        "version": settings.version,
        "active_sessions": manager.active_count(),
        "environment": settings.environment,
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics scraping endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
