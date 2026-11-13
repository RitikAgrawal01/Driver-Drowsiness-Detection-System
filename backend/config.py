"""
backend/config.py — INFRASTRUCTURE settings only (from environment / .env).

Everything that changes behaviour (fps, window length, alarm timing, drift thresholds) comes
from params.yaml via ddd_core.config.load_params(); the window length and decision threshold
come from the registered model itself. One source of truth for each setting.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "Driver Drowsiness Detection System"
    environment: str = "development"
    version: str = "2.0.0"
    log_level: str = "INFO"

    model_server_url: str = "http://localhost:8001"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Airflow REST API (read-only use by /api/pipeline/*; credentials stay server-side)
    airflow_url: str = "http://localhost:8080"
    airflow_user: str = ""
    airflow_password: str = ""

    # Comma-separated list of browser origins allowed to call the API (no wildcard).
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")

    model_config = {"env_file": ".env", "extra": "ignore", "case_sensitive": False, "protected_namespaces": ()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip() and o.strip() != "*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
