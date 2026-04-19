"""
backend/services/model_client.py
──────────────────────────────────
Async HTTP client that calls the model server's /predict endpoint.
Uses httpx with connection pooling and timeout handling.
"""

import time
from typing import Optional

import httpx

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import MODEL_SERVER_REACHABLE, REQUEST_ERRORS_TOTAL

logger = get_logger("model_client")

# Module-level client (reused across requests)
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        _client = httpx.AsyncClient(
            base_url=settings.model_server_url,
            timeout=httpx.Timeout(
                connect=2.0,
                read=settings.model_server_timeout_sec,
                write=2.0,
                pool=1.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
    return _client


async def close_client():
    """Close the HTTP client — call on app shutdown."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def predict(features: dict) -> dict:
    """
    POST feature vector to model server /predict.

    Args:
        features: dict matching FeatureVector schema

    Returns:
        dict with state, confidence, inference_latency_ms, model_version

    Raises:
        httpx.HTTPError: if model server is unreachable
        ValueError: if model server returns unexpected response
    """
    client = get_client()
    t_start = time.perf_counter()

    try:
        response = await client.post("/predict", json=features)
        response.raise_for_status()

        data = response.json()
        MODEL_SERVER_REACHABLE.set(1)

        logger.debug(
            "Model server prediction",
            state=data.get("state"),
            confidence=data.get("confidence"),
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
        )
        return data

    except httpx.TimeoutException as e:
        MODEL_SERVER_REACHABLE.set(0)
        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/predict", error_type="timeout"
        ).inc()
        logger.error(f"Model server timeout: {e}")
        raise

    except httpx.ConnectError as e:
        MODEL_SERVER_REACHABLE.set(0)
        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/predict", error_type="connection_error"
        ).inc()
        logger.error(f"Cannot connect to model server: {e}")
        raise

    except httpx.HTTPStatusError as e:
        MODEL_SERVER_REACHABLE.set(0)
        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/predict", error_type=f"http_{e.response.status_code}"
        ).inc()
        logger.error(
            f"Model server HTTP error: {e.response.status_code}",
            body=e.response.text,
        )
        raise

    except Exception as e:
        MODEL_SERVER_REACHABLE.set(0)
        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/predict", error_type="unknown"
        ).inc()
        logger.exception(f"Unexpected model client error: {e}")
        raise


async def health_check() -> bool:
    """
    Check if model server /ready endpoint responds.
    Used by backend /ready endpoint and periodic health probe.
    """
    try:
        client = get_client()
        r = await client.get("/ready", timeout=2.0)
        reachable = r.status_code == 200
        MODEL_SERVER_REACHABLE.set(1 if reachable else 0)
        return reachable
    except Exception as e:
        # ── ADDED DEBUG PRINTS ──
        settings = get_settings()
        print(f"\n❌ DEBUG FAILED: Trying to reach {settings.model_server_url}/ready")
        print(f"❌ ERROR EXACT: {type(e).__name__} - {str(e)}\n")
        # ────────────────────────
        MODEL_SERVER_REACHABLE.set(0)
        return False
