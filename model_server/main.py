"""
model_server/main.py — FastAPI inference service for the registered champion model.

Endpoints: /health (process up), /ready (model loaded), /predict, /model/info, /metrics,
/admin/reload (hot-swap, token-protected when ADMIN_TOKEN is set).
Internal service: only the backend calls it, so no CORS.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from model_server.admin import router as admin_router
from model_server.model_loader import ModelNotLoaded, store
from model_server.schemas import FeatureVector, PredictionResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("model_server")

INFERENCE_LATENCY = Histogram(
    "ddd_model_inference_latency_seconds",
    "Model-only inference latency",
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
PREDICTION_SCORE = Histogram(
    "ddd_model_prediction_score",
    "Model score (higher = more drowsy)",
    buckets=[-2, -1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2],
)
REQUESTS = Counter("ddd_model_requests_total", "Prediction requests", ["status"])
DROWSY_PREDICTIONS = Counter("ddd_model_drowsy_predictions_total", "Drowsy predictions")
MODEL_LOADED = Gauge("ddd_model_loaded", "1 if a model is loaded")
MODEL_VERSION = Gauge("ddd_model_version", "Registered version being served")


def try_load() -> bool:
    try:
        meta = store.load()
        MODEL_LOADED.set(1)
        MODEL_VERSION.set(float(meta["model_version"]))
        return True
    except Exception as e:  # stay up, report not-ready, keep any previous model
        logger.error("Model load failed: %s", e)
        MODEL_LOADED.set(1 if store.is_loaded() else 0)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try_load()
    yield


app = FastAPI(title="DDD Model Server", version="2.0.0", lifespan=lifespan)
app.include_router(admin_router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "model_server"}


@app.get("/ready", tags=["Health"])
async def ready():
    if not store.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", **store.meta()}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: FeatureVector):
    try:
        result = store.predict(features.model_dump())
    except ModelNotLoaded:
        REQUESTS.labels(status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        REQUESTS.labels(status="500").inc()
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    REQUESTS.labels(status="200").inc()
    INFERENCE_LATENCY.observe(result["inference_latency_ms"] / 1000)
    PREDICTION_SCORE.observe(result["score"])
    if result["state"] == "drowsy":
        DROWSY_PREDICTIONS.inc()
    return result


@app.get("/model/info", tags=["Model"])
async def model_info():
    if not store.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return store.meta()


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
