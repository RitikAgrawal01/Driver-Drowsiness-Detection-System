"""
backend/config.py
──────────────────
Centralised settings loaded from environment variables / .env file.
All values have sensible defaults so the app starts without a .env file.
Pydantic-settings validates types and ranges at startup.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Service identity ──────────────────────────────────────────────────────
    project_name: str = "Driver Drowsiness Detection System"
    environment: str = "development"
    version: str = "1.0.0"

    # ── Backend server ────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"

    # ── Model server ──────────────────────────────────────────────────────────
    model_server_url: str = "http://localhost:8001"
    model_server_timeout_sec: float = 5.0

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # ── Feature engineering ───────────────────────────────────────────────────
    sliding_window_size: int = Field(default=30, ge=5, le=300)
    ear_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    mar_threshold: float = Field(default=0.60, ge=0.0)
    perclos_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    frame_rate: int = Field(default=30, ge=1, le=120)

    # ── Inference / alerting ──────────────────────────────────────────────────
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    inference_latency_target_ms: float = 200.0

    # ── Drift detection ───────────────────────────────────────────────────────
    drift_score_threshold: float = Field(default=0.15, ge=0.0)
    baseline_json_path: str = "data/features/baseline.json"
    drift_window_size: int = Field(default=300, ge=50)  # frames for live dist

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list = ["http://localhost:3000", "http://localhost:3001"]

    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — call get_settings() everywhere."""
    return Settings()
