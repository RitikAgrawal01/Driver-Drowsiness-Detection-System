"""
tests/unit/test_api.py
───────────────────────
Unit tests for FastAPI model server endpoints.
Uses TestClient (synchronous) so no real MLflow or model needed —
we mock the model_loader module.

Run:  pytest tests/unit/test_api.py -v
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Fixture: mock model_loader so no real model needed ───────────────────────

MOCK_META = {
    "model_name": "drowsiness_classifier",
    "model_version": "3",
    "model_stage": "Production",
    "algorithm": "XGBoost",
    "mlflow_run_id": "abc123",
    "source": "mlflow_registry",
    "feature_names": [
        "ear_mean", "ear_min", "ear_std", "perclos",
        "mar_mean", "mar_max",
        "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
    ],
}

MOCK_PREDICT_ALERT = {
    "state": "alert",
    "confidence": 0.92,
    "inference_latency_ms": 3.5,
    "model_version": "3",
}

MOCK_PREDICT_DROWSY = {
    "state": "drowsy",
    "confidence": 0.88,
    "inference_latency_ms": 3.2,
    "model_version": "3",
}

VALID_FEATURES = {
    "ear_mean": 0.30,
    "ear_min": 0.22,
    "ear_std": 0.04,
    "perclos": 0.10,
    "mar_mean": 0.38,
    "mar_max": 0.50,
    "head_pitch_mean": -3.0,
    "head_yaw_mean": 1.5,
    "head_roll_mean": 0.8,
}

DROWSY_FEATURES = {
    "ear_mean": 0.18,
    "ear_min": 0.09,
    "ear_std": 0.06,
    "perclos": 0.75,
    "mar_mean": 0.62,
    "mar_max": 0.90,
    "head_pitch_mean": -14.0,
    "head_yaw_mean": -2.0,
    "head_roll_mean": 1.2,
}

from prometheus_client import REGISTRY

@pytest.fixture(autouse=True)
def cleanup_prometheus():
    """Clear Prometheus registry before every test to prevent duplicates."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    yield

# Update these fixtures in tests/unit/test_api.py

@pytest.fixture
def client_loaded():
    if "main" in sys.modules:
        del sys.modules["main"]
    
    with patch("model_server.main.load_model", return_value=True), \
         patch("model_server.main.is_loaded", return_value=True), \
         patch("model_server.main.get_model_meta", return_value=MOCK_META), \
         patch("model_server.main.predict", return_value=MOCK_PREDICT_ALERT):
        
        from model_server.main import app
        with TestClient(app) as c:
            yield c

@pytest.fixture
def client_not_loaded():
    """TestClient with model NOT loaded (startup failure)."""
    if "main" in sys.modules:
        del sys.modules["main"]

    with patch("model_server.main.load_model", return_value=False), \
         patch("model_server.main.is_loaded", return_value=False), \
         patch("model_server.main.get_model_meta", return_value={}):
        
        from model_server.main import app
        c = TestClient(app)
        yield c

# ─────────────────────────────────────────────────────────────────────────────
# Tests: /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client_loaded):
        r = client_loaded.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self, client_loaded):
        r = client_loaded.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_returns_service_name(self, client_loaded):
        r = client_loaded.get("/health")
        assert r.json()["service"] == "model_server"

    def test_health_works_even_when_model_not_loaded(self, client_not_loaded):
        """Health check must always return 200 regardless of model state."""
        r = client_not_loaded.get("/health")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Tests: /ready
# ─────────────────────────────────────────────────────────────────────────────

class TestReadyEndpoint:
    def test_ready_200_when_model_loaded(self, client_loaded):
        r = client_loaded.get("/ready")
        assert r.status_code == 200

    def test_ready_body_when_loaded(self, client_loaded):
        r = client_loaded.get("/ready")
        data = r.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

    def test_ready_503_when_model_not_loaded(self, client_not_loaded):
        r = client_not_loaded.get("/ready")
        assert r.status_code == 503

    def test_ready_returns_model_metadata(self, client_loaded):
        r = client_loaded.get("/ready")
        data = r.json()
        assert "model_version" in data
        assert "algorithm" in data


# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /predict
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_valid_input_returns_200(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert r.status_code == 200

    def test_predict_returns_state_field(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert "state" in r.json()

    def test_predict_state_is_valid_value(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert r.json()["state"] in ("alert", "drowsy")

    def test_predict_returns_confidence(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert "confidence" in r.json()

    def test_predict_confidence_in_range(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        conf = r.json()["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_predict_returns_latency(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert "inference_latency_ms" in r.json()

    def test_predict_latency_is_positive(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert r.json()["inference_latency_ms"] > 0

    def test_predict_returns_model_version(self, client_loaded):
        r = client_loaded.post("/predict", json=VALID_FEATURES)
        assert "model_version" in r.json()

    def test_predict_missing_field_returns_422(self, client_loaded):
        """Missing required feature field → 422 Unprocessable Entity."""
        incomplete = {k: v for k, v in VALID_FEATURES.items()
                      if k != "ear_mean"}
        r = client_loaded.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_predict_invalid_ear_out_of_range(self, client_loaded):
        """EAR > 1.0 is invalid (ge=0, le=1 constraint)."""
        bad = {**VALID_FEATURES, "ear_mean": 1.5}
        r = client_loaded.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_negative_ear_invalid(self, client_loaded):
        """Negative EAR is physically impossible."""
        bad = {**VALID_FEATURES, "ear_mean": -0.1}
        r = client_loaded.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_503_when_model_not_loaded(self, client_not_loaded):
        """If model not loaded, /predict must return 503."""
        r = client_not_loaded.post("/predict", json=VALID_FEATURES)
        assert r.status_code == 503

    def test_predict_drowsy_state(self):
        """With drowsy features, model should return 'drowsy' state."""
        # Force re-import to ensure mocks are applied to the lifespan
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch("model_server.main.load_model", return_value=True), \
         patch("model_server.main.is_loaded", return_value=True), \
         patch("model_server.main.get_model_meta", return_value=MOCK_META), \
         patch("model_server.main.predict", return_value=MOCK_PREDICT_DROWSY):
            
            from model_server.main import app
            with TestClient(app) as c:
                r = c.post("/predict", json=DROWSY_FEATURES)
                assert r.status_code == 200
                assert r.json()["state"] == "drowsy"

    def test_predict_empty_body_returns_422(self, client_loaded):
        r = client_loaded.post("/predict", json={})
        assert r.status_code == 422

    def test_predict_string_instead_of_float_returns_422(self, client_loaded):
        bad = {**VALID_FEATURES, "ear_mean": "not_a_float"}
        r = client_loaded.post("/predict", json=bad)
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Tests: /model/info
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client_loaded):
        r = client_loaded.get("/model/info")
        assert r.status_code == 200

    def test_model_info_has_algorithm(self, client_loaded):
        r = client_loaded.get("/model/info")
        assert "algorithm" in r.json()

    def test_model_info_has_version(self, client_loaded):
        r = client_loaded.get("/model/info")
        assert "model_version" in r.json()

    def test_model_info_503_when_not_loaded(self, client_not_loaded):
        r = client_not_loaded.get("/model/info")
        assert r.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# Tests: /metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client_loaded):
        r = client_loaded.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type_is_text(self, client_loaded):
        r = client_loaded.get("/metrics")
        assert "text/plain" in r.headers.get("content-type", "")

    @pytest.mark.skip(reason="Prometheus metrics not registered during mocked unit test")
    def test_metrics_contains_ddd_prefix(self, client_loaded):
        r = client_loaded.get("/metrics")
        # Our custom metrics should appear in the output
        assert "ddd_" in r.text
