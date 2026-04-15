"""
model_server/main.py
MLflow-backed model inference server.
Full implementation in Phase 5.
"""
from fastapi import FastAPI

app = FastAPI(
    title="Driver Drowsiness Detection — Model Server",
    description="Loads the production model from MLflow registry and exposes /predict.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check — required for Docker healthcheck."""
    return {"status": "ok", "service": "model_server"}


@app.get("/ready")
async def ready():
    """Readiness — ok only when model is loaded from MLflow registry."""
    return {"status": "ready", "model_loaded": False}  # updated in Phase 5
