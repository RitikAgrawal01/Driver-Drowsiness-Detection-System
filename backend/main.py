"""
backend/main.py
FastAPI main application entry point.
Full implementation in Phase 5.
"""
from fastapi import FastAPI

app = FastAPI(
    title="Driver Drowsiness Detection — Backend API",
    description="Main API for drowsiness detection. Receives webcam frames, computes features, calls model server.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint — required for Docker healthcheck and /ready pattern."""
    return {"status": "ok", "service": "backend"}


@app.get("/ready")
async def ready():
    """Readiness check — returns ok only when model server is reachable."""
    return {"status": "ready"}
