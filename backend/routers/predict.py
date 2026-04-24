"""
backend/routers/predict.py
───────────────────────────
POST /predict — REST endpoint for batch/test inference.
Real-time inference uses the WebSocket endpoint in stream.py.
"""

import time

from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import (
    INFERENCE_LATENCY, PREDICTION_CONFIDENCE,
    REQUESTS_TOTAL, REQUEST_ERRORS_TOTAL, DROWSY_ALERTS_TOTAL,
)
from backend.schemas import PredictRequest, PredictionResponse
from backend.services import model_client
from backend.services.drift_detector import get_global_detector

logger = get_logger("router.predict")
router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """
    Accept a precomputed feature vector and return a drowsiness prediction.
    Used for batch inference and testing.
    Real-time sessions use WebSocket /ws instead.
    """
    settings = get_settings()
    t_start = time.perf_counter()

    try:
        result = await model_client.predict(request.features.model_dump())
    except Exception as e:
        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/predict", error_type="model_server_error"
        ).inc()
        raise HTTPException(
            status_code=503,
            detail="Model server unavailable. Please try again shortly.",
        )

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Prometheus
    INFERENCE_LATENCY.observe(latency_ms / 1000)
    PREDICTION_CONFIDENCE.observe(result.get("confidence", 0))
    REQUESTS_TOTAL.labels(endpoint="/predict", status="200").inc()

    alert_triggered = (
        result.get("state") == "drowsy"
        and result.get("confidence", 0) >= settings.confidence_threshold
    )
    if alert_triggered:
        DROWSY_ALERTS_TOTAL.inc()

    # Update global drift detector
    detector = get_global_detector()
    detector.update(request.features.model_dump())
    detector.compute_drift_scores()
    
    logger.info(
        "Prediction",
        state=result.get("state"),
        confidence=result.get("confidence"),
        latency_ms=round(latency_ms, 2),
        alert=alert_triggered,
    )

    return PredictionResponse(
        state=result["state"],
        confidence=result["confidence"],
        inference_latency_ms=round(latency_ms, 2),
        model_name="drowsiness_classifier",
        model_version=result.get("model_version", "unknown"),
        alert_triggered=alert_triggered,
    )
