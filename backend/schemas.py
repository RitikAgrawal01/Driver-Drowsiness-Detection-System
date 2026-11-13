"""Backend API schemas. The feature contract is shared with the model server (ddd_core.schemas)."""

from typing import Literal

from pydantic import BaseModel

from ddd_core.schemas import FeatureVector


class PredictRequest(BaseModel):
    features: FeatureVector


class PredictResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    score: float
    threshold: float
    model_version: str
    model_type: str | None = None
    window_s: int
    inference_latency_ms: float

    model_config = {"protected_namespaces": ()}
