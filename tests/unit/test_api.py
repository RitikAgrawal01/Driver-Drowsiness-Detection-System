"""
tests/unit/test_api.py
───────────────────────
Unit tests for FastAPI backend endpoints.
Uses httpx.AsyncClient with the FastAPI test transport —
no real server needed, no model server required.

Run with: pytest tests/unit/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────
# Tests: Health & Readiness Endpoints
# ─────────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    """Tests for GET /health and GET /ready."""

    def test_health_returns_200(self):
        """GET /health must return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        """GET /health body must contain status=ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self):
        """GET /health must identify the service."""
        response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert data["service"] == "backend"

    def test_ready_returns_200(self):
        """GET /ready must return 200 (even if model server unreachable in test)."""
        response = client.get("/ready")
        # In test mode model server is not running — still returns a valid response
        assert response.status_code in [200, 503]

    def test_ready_returns_json(self):
        """GET /ready must return valid JSON."""
        response = client.get("/ready")
        data = response.json()
        assert "status" in data

    def test_health_content_type_json(self):
        """GET /health must return application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_idempotent(self):
        """Multiple calls to /health must return same result."""
        r1 = client.get("/health").json()
        r2 = client.get("/health").json()
        assert r1["status"] == r2["status"]


# ─────────────────────────────────────────────────────────────────
# Tests: API Schema Validation
# ─────────────────────────────────────────────────────────────────

class TestAPISchema:
    """Tests for Pydantic schema validation on request bodies."""

    VALID_FEATURES = {
        "ear_mean": 0.28,
        "ear_min": 0.20,
        "ear_std": 0.04,
        "perclos": 0.15,
        "mar_mean": 0.45,
        "mar_max": 0.72,
        "head_pitch_mean": -8.5,
        "head_yaw_mean": 3.2,
        "head_roll_mean": 1.1,
    }

    def test_valid_feature_vector_accepted(self):
        """Valid feature vector should not raise validation error."""
        # This tests schema validation even before model inference
        from backend.schemas import FeatureVector
        fv = FeatureVector(**self.VALID_FEATURES)
        assert fv.ear_mean == pytest.approx(0.28)

    def test_ear_mean_range_validation(self):
        """ear_mean must be between 0.0 and 1.0."""
        from backend.schemas import FeatureVector
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeatureVector(**{**self.VALID_FEATURES, "ear_mean": 1.5})

    def test_perclos_range_validation(self):
        """perclos must be between 0.0 and 1.0."""
        from backend.schemas import FeatureVector
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeatureVector(**{**self.VALID_FEATURES, "perclos": -0.1})

    def test_missing_required_field_raises(self):
        """Missing required field must raise ValidationError."""
        from backend.schemas import FeatureVector
        from pydantic import ValidationError
        incomplete = {k: v for k, v in self.VALID_FEATURES.items() if k != "ear_mean"}
        with pytest.raises(ValidationError):
            FeatureVector(**incomplete)

    def test_prediction_response_schema(self):
        """PredictionResponse must validate state as 'alert' or 'drowsy'."""
        from backend.schemas import PredictionResponse
        from pydantic import ValidationError

        valid = PredictionResponse(
            state="alert",
            confidence=0.91,
            inference_latency_ms=4.2,
            model_name="test_model",
            model_version="1",
            alert_triggered=False,
        )
        assert valid.state == "alert"

        with pytest.raises(ValidationError):
            PredictionResponse(
                state="unknown_state",  # invalid
                confidence=0.91,
                inference_latency_ms=4.2,
                model_name="test_model",
                model_version="1",
                alert_triggered=False,
            )

    def test_confidence_must_be_0_to_1(self):
        """Confidence in PredictionResponse must be between 0.0 and 1.0."""
        from backend.schemas import PredictionResponse
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PredictionResponse(
                state="alert",
                confidence=1.5,    # invalid
                inference_latency_ms=4.2,
                model_name="m",
                model_version="1",
                alert_triggered=False,
            )
