"""The one feature-vector contract shared by the backend and the model server.
Ranges reject impossible inputs at the API boundary (HTTP 422)."""

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """The 12 window features produced by ddd_core.windows.window_features."""

    ear_mean: float = Field(..., ge=0.0, le=1.0)
    ear_min: float = Field(..., ge=0.0, le=1.0)
    ear_std: float = Field(..., ge=0.0, le=1.0)
    perclos: float = Field(..., ge=0.0, le=1.0)
    mar_mean: float = Field(..., ge=0.0, le=3.0)
    mar_max: float = Field(..., ge=0.0, le=3.0)
    head_pitch_mean: float = Field(..., ge=-180.0, le=180.0)
    head_yaw_mean: float = Field(..., ge=-180.0, le=180.0)
    head_roll_mean: float = Field(..., ge=-180.0, le=180.0)
    blink_rate_pm: float = Field(..., ge=0.0, le=600.0)
    blink_dur_mean_s: float = Field(..., ge=0.0, le=60.0)
    long_closure_rate_pm: float = Field(..., ge=0.0, le=600.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "ear_mean": 0.28,
                "ear_min": 0.12,
                "ear_std": 0.04,
                "perclos": 0.05,
                "mar_mean": 0.02,
                "mar_max": 0.30,
                "head_pitch_mean": -3.0,
                "head_yaw_mean": 2.0,
                "head_roll_mean": 1.0,
                "blink_rate_pm": 14.0,
                "blink_dur_mean_s": 0.2,
                "long_closure_rate_pm": 0.0,
            }
        }
    }
