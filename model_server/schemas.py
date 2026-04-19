"""
model_server/schemas.py
────────────────────────
Shared Pydantic models for model server request/response validation.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]


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


class PredictionResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_latency_ms: float = Field(..., ge=0.0)
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "state": "alert",
                "confidence": 0.93,
                "inference_latency_ms": 4.2,
                "model_version": "3",
            }
        }


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    algorithm: str
    mlflow_run_id: str
    source: str
    feature_names: list
    metrics: Optional[Dict[str, float]] = None
