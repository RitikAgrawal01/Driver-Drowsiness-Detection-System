"""
backend/main.py
────────────────
FastAPI main application entry point for the DDD backend.
Full router implementation done in Phase 5.
Health and ready endpoints are fully implemented here.
"""

import logging
import os

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import HealthResponse, ReadyResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backend")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Driver Drowsiness Detection — Backend API",
    description=(
        "Main API for drowsiness detection. "
        "Receives webcam frames via WebSocket, computes MediaPipe features, "
        "calls model server for inference, and streams predictions back."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow React dev server and production frontend) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://frontend:80",
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model_server:8001")


# ── Health & Ready Endpoints ──────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Infrastructure"],
    summary="Service health check",
)
async def health():
    """
    Health check endpoint.
    Used by Docker Compose healthcheck and load balancers.
    Always returns 200 OK if the process is running.
    """
    return HealthResponse(status="ok", service="backend", version="1.0.0")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["Infrastructure"],
    summary="Service readiness check",
)
async def ready():
    """
    Readiness check.
    Returns ok only when the model server /health endpoint is reachable.
    Frontend uses this to show "connecting..." state on startup.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{MODEL_SERVER_URL}/health")
            reachable = resp.status_code == 200
    except Exception as e:
        logger.warning(f"Model server not reachable: {e}")
        reachable = False

    if reachable:
        return ReadyResponse(
            status="ready",
            model_server_reachable=True,
            model_server_url=MODEL_SERVER_URL,
        )
    else:
        from fastapi import Response
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(
                status="not_ready",
                model_server_reachable=False,
                model_server_url=MODEL_SERVER_URL,
                detail="Model server health check failed",
            ).model_dump(),
        )


# ── Placeholder routers (implemented Phase 5) ─────────────────────────────────
# from routers import predict, session, stream
# app.include_router(predict.router, prefix="/predict", tags=["Inference"])
# app.include_router(session.router, prefix="/session", tags=["Session"])
# app.include_router(stream.router, tags=["WebSocket"])
