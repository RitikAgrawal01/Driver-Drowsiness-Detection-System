"""
backend/schemas.py
───────────────────
All Pydantic request/response models for the backend API.
These define the strict I/O contracts documented in the LLD.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ── Feature vector ────────────────────────────────────────────────────────────

class FeatureVector(BaseModel):
    ear_mean: float = Field(..., ge=0.0, le=1.0)
    ear_min: float = Field(..., ge=0.0, le=1.0)
    ear_std: float = Field(..., ge=0.0)
    perclos: float = Field(..., ge=0.0, le=1.0)
    mar_mean: float = Field(..., ge=0.0)
    mar_max: float = Field(..., ge=0.0)
    head_pitch_mean: float
    head_yaw_mean: float
    head_roll_mean: float

    def to_list(self) -> list:
        return [
            self.ear_mean, self.ear_min, self.ear_std,
            self.perclos, self.mar_mean, self.mar_max,
            self.head_pitch_mean, self.head_yaw_mean, self.head_roll_mean,
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "ear_mean": 0.28, "ear_min": 0.19, "ear_std": 0.04,
                "perclos": 0.20, "mar_mean": 0.45, "mar_max": 0.72,
                "head_pitch_mean": -8.5, "head_yaw_mean": 3.2,
                "head_roll_mean": 1.1,
            }
        }


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: Optional[str] = None
    features: FeatureVector


class PredictionResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_latency_ms: float
    model_name: str = "drowsiness_classifier"
    model_version: str
    alert_triggered: bool


# ── Session ───────────────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    driver_id: Optional[str] = None
    window_size: int = Field(default=30, ge=10, le=300)

    class Config:
        json_schema_extra = {
            "example": {"driver_id": "driver_001", "window_size": 30}
        }


class SessionStartResponse(BaseModel):
    session_id: str
    started_at: str
    window_size: int
    status: str = "active"


class SessionStopRequest(BaseModel):
    session_id: str


class SessionStopResponse(BaseModel):
    session_id: str
    started_at: str
    ended_at: str
    duration_seconds: float
    total_frames_processed: int
    drowsy_alerts_triggered: int
    average_ear: Optional[float]
    average_confidence: Optional[float]
    status: str = "completed"


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    frames_processed: int
    current_state: Optional[str]
    current_confidence: Optional[float]
    current_ear: Optional[float]
    current_mar: Optional[float]
    current_perclos: Optional[float]
    alerts_triggered: int
    drift_score: float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    model_server_reachable: bool
    model_server_url: str
