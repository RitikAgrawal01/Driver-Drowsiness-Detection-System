"""Model-server response schema. The request schema is the shared ddd_core.schemas.FeatureVector."""

from typing import Literal

from pydantic import BaseModel, Field

from ddd_core.schemas import FeatureVector  # noqa: F401  (re-exported)


class PredictionResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    score: float
    threshold: float
    model_version: str
    model_type: str | None
    window_s: int
    inference_latency_ms: float = Field(..., ge=0.0)

    model_config = {"protected_namespaces": ()}
