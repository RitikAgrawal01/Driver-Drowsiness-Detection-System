"""
backend/config.py
Centralised settings loaded from environment variables / .env file.
Uses pydantic-settings for validation.
Full implementation in Phase 5.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_server_url: str = "http://model_server:8001"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    log_level: str = "INFO"
    sliding_window_size: int = 30
    confidence_threshold: float = 0.7
    drift_score_threshold: float = 0.15

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
