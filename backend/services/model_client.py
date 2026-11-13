"""Async HTTP client for the model server (pooled connections, explicit timeouts).
Errors propagate as ModelServerError; callers decide the user-facing state (never "alert")."""

import time

import httpx

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import MODEL_CALL_LATENCY, MODEL_SERVER_REACHABLE
from ddd_core.config import load_params

logger = get_logger("model_client")
_client: httpx.AsyncClient | None = None


class ModelServerError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        timeout = float(load_params()["serving"]["model_timeout_s"])
        _client = httpx.AsyncClient(
            base_url=get_settings().model_server_url,
            timeout=httpx.Timeout(timeout, connect=2.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def predict(features: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = await get_client().post("/predict", json=features)
    except httpx.TimeoutException as e:
        MODEL_SERVER_REACHABLE.set(0)
        raise ModelServerError("timeout", str(e)) from e
    except httpx.HTTPError as e:
        MODEL_SERVER_REACHABLE.set(0)
        raise ModelServerError("connection_error", str(e)) from e
    finally:
        MODEL_CALL_LATENCY.observe(time.perf_counter() - t0)
    if r.status_code != 200:
        if r.status_code == 503:
            MODEL_SERVER_REACHABLE.set(0)
        raise ModelServerError(f"http_{r.status_code}", r.text[:300])
    MODEL_SERVER_REACHABLE.set(1)
    return r.json()


async def model_info() -> dict | None:
    """Metadata of the model being served, or None if the model server is not ready."""
    try:
        r = await get_client().get("/model/info", timeout=3.0)
    except httpx.HTTPError as e:
        logger.warning("model server unreachable", error=str(e))
        MODEL_SERVER_REACHABLE.set(0)
        return None
    MODEL_SERVER_REACHABLE.set(1 if r.status_code == 200 else 0)
    return r.json() if r.status_code == 200 else None
