"""
backend/schemas.py
───────────────────
All Pydantic request/response models for the DDD backend API.
These are the single source of truth for all API I/O schemas —
referenced by LLD.md and used by FastAPI for automatic validation.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


# ── Feature Vector ────────────────────────────────────────────────────────────

class FeatureVector(BaseModel):
    """
    Sliding-window feature vector for drowsiness inference.
    One instance represents features computed over SLIDING_WINDOW_SIZE frames.
    """
    ear_mean: float = Field(..., ge=0.0, le=1.0, description="Mean Eye Aspect Ratio over window")
    ear_min: float = Field(..., ge=0.0, le=1.0, description="Minimum EAR (captures worst blink)")
    ear_std: float = Field(..., ge=0.0, description="EAR standard deviation")
    perclos: float = Field(..., ge=0.0, le=1.0, description="Fraction of frames with EAR below threshold")
    mar_mean: float = Field(..., ge=0.0, description="Mean Mouth Aspect Ratio (yawn detection)")
    mar_max: float = Field(..., ge=0.0, description="Maximum MAR in window")
    head_pitch_mean: float = Field(..., description="Mean head pitch in degrees (negative = nodding down)")
    head_yaw_mean: float = Field(..., description="Mean head yaw in degrees")
    head_roll_mean: float = Field(..., description="Mean head roll in degrees")

    class Config:
        json_schema_extra = {
            "example": {
                "ear_mean": 0.28,
                "ear_min": 0.19,
                "ear_std": 0.04,
                "perclos": 0.20,
                "mar_mean": 0.45,
                "mar_max": 0.72,
                "head_pitch_mean": -8.5,
                "head_yaw_mean": 3.2,
                "head_roll_mean": 1.1,
            }
        }


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for POST /predict."""
    session_id: Optional[str] = Field(None, description="Active session ID (optional for batch use)")
    features: FeatureVector


class PredictionResponse(BaseModel):
    """Response body for POST /predict and model server POST /predict."""
    state: Literal["alert", "drowsy"] = Field(..., description="Predicted drowsiness state")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    inference_latency_ms: float = Field(..., ge=0.0, description="End-to-end inference time in ms")
    model_name: str = Field(..., description="Name of the model used for inference")
    model_version: str = Field(..., description="MLflow model registry version")
    alert_triggered: bool = Field(..., description="True if confidence exceeds alert threshold")


# ── Session ───────────────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    """Request body for POST /session/start."""
    driver_id: Optional[str] = Field(None, description="Optional driver identifier")
    window_size: int = Field(default=30, ge=10, le=300, description="Sliding window size in frames")


class SessionStartResponse(BaseModel):
    """Response body for POST /session/start."""
    session_id: str
    started_at: str
    window_size: int
    status: Literal["active"]


class SessionStopRequest(BaseModel):
    """Request body for POST /session/stop."""
    session_id: str


class SessionStopResponse(BaseModel):
    """Response body for POST /session/stop."""
    session_id: str
    started_at: str
    ended_at: str
    duration_seconds: float
    total_frames_processed: int
    drowsy_alerts_triggered: int
    average_ear: float
    average_confidence: float
    status: Literal["completed"]


class SessionStatusResponse(BaseModel):
    """Response body for GET /session/{session_id}/status."""
    session_id: str
    status: Literal["active", "completed", "error"]
    frames_processed: int
    current_state: Optional[Literal["alert", "drowsy"]] = None
    current_confidence: Optional[float] = None
    current_ear: Optional[float] = None
    current_mar: Optional[float] = None
    current_perclos: Optional[float] = None
    alerts_triggered: int
    drift_score: float


# ── Health / Ready ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_server_reachable: bool
    model_server_url: Optional[str] = None
    detail: Optional[str] = None
