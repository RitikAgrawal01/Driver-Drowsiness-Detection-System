"""POST /predict — score one precomputed feature vector (batch use and testing).
Live monitoring uses the WebSocket in stream.py. Errors are 503, never a default prediction."""

from fastapi import APIRouter, HTTPException

from backend.metrics import PREDICTION_ERRORS, PREDICTION_SCORE, PREDICTIONS
from backend.schemas import PredictRequest, PredictResponse
from backend.services import model_client

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    PREDICTIONS.labels(source="rest").inc()
    try:
        result = await model_client.predict(request.features.model_dump())
    except model_client.ModelServerError as e:
        PREDICTION_ERRORS.labels(source="rest", error_type=e.error_type).inc()
        status = 422 if e.error_type == "http_422" else 503
        raise HTTPException(status_code=status, detail=f"model server error: {e.error_type}")
    PREDICTION_SCORE.observe(result["score"])
    return PredictResponse(**result)
