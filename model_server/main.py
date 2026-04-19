"""
model_server/main.py
─────────────────────
FastAPI inference server.
Loads the Production model from MLflow registry on startup.
Exposes /predict, /health, /ready, /model/info, /metrics.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import load_model, get_model_meta, is_loaded, predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("model_server")

# ── Prometheus metrics ────────────────────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "ddd_model_inference_latency_seconds",
    "Model-only inference latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "ddd_model_prediction_confidence",
    "Prediction confidence from model",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
REQUEST_COUNTER = Counter(
    "ddd_model_requests_total",
    "Total prediction requests to model server",
    ["status"],
)
DROWSY_PREDICTIONS = Counter(
    "ddd_model_drowsy_predictions_total",
    "Total drowsy predictions made",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class FeatureVector(BaseModel):
    ear_mean: float = Field(..., ge=0.0, le=1.0,
                             description="Mean Eye Aspect Ratio over window")
    ear_min: float = Field(..., ge=0.0, le=1.0,
                            description="Minimum EAR in window")
    ear_std: float = Field(..., ge=0.0,
                            description="EAR standard deviation")
    perclos: float = Field(..., ge=0.0, le=1.0,
                            description="Fraction of frames where EAR < threshold")
    mar_mean: float = Field(..., ge=0.0,
                             description="Mean Mouth Aspect Ratio")
    mar_max: float = Field(..., ge=0.0,
                            description="Maximum MAR in window")
    head_pitch_mean: float = Field(...,
                                    description="Mean head pitch angle (degrees)")
    head_yaw_mean: float = Field(...,
                                  description="Mean head yaw angle (degrees)")
    head_roll_mean: float = Field(...,
                                   description="Mean head roll angle (degrees)")

    def to_list(self) -> list:
        return [
            self.ear_mean, self.ear_min, self.ear_std,
            self.perclos,
            self.mar_mean, self.mar_max,
            self.head_pitch_mean, self.head_yaw_mean, self.head_roll_mean,
        ]


class PredictionResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    confidence: float
    inference_latency_ms: float
    model_version: str


# ── App lifespan (startup / shutdown) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Model server starting — loading model...")
    success = load_model()
    if success:
        meta = get_model_meta()
        logger.info(
            f"✓ Model ready: {meta['algorithm']} "
            f"v{meta['model_version']} ({meta['source']})"
        )
    else:
        logger.error("✗ Model failed to load! /predict will return 503.")
    yield
    logger.info("Model server shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DDD Model Server",
    description="Drowsiness Detection inference server. Loads Production model from MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    """Docker healthcheck endpoint."""
    return {"status": "ok", "service": "model_server"}


@app.get("/ready", tags=["Health"])
async def ready():
    """Readiness — ok only when model is loaded."""
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not yet loaded from MLflow registry",
        )
    meta = get_model_meta()
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "model_stage": meta["model_stage"],
        "algorithm": meta["algorithm"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_endpoint(features: FeatureVector):
    """
    Run drowsiness inference on a precomputed feature vector.
    Called exclusively by the backend service — frontend never calls this directly.
    """
    if not is_loaded():
        REQUEST_COUNTER.labels(status="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service not ready.",
        )

    try:
        result = predict(features.to_list())

        # Record Prometheus metrics
        INFERENCE_LATENCY.observe(result["inference_latency_ms"] / 1000)
        PREDICTION_CONFIDENCE.observe(result["confidence"])
        REQUEST_COUNTER.labels(status="200").inc()

        if result["state"] == "drowsy":
            DROWSY_PREDICTIONS.inc()

        return PredictionResponse(**result)

    except Exception as e:
        REQUEST_COUNTER.labels(status="500").inc()
        logger.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Returns metadata about the currently loaded model."""
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return get_model_meta()


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint — scraped by Prometheus every 10s."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
