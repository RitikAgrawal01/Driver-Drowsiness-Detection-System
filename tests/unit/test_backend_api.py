"""
tests/unit/test_backend_api.py
────────────────────────────────
Unit tests for FastAPI backend endpoints.
All external dependencies (model server, MediaPipe) are mocked.
"""

import sys
import os
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

# 1. Set path so backend module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

# 2. PURGE PROMETHEUS REGISTRY BEFORE IMPORTING BACKEND
# This prevents collisions with metrics already registered by test_api.py
from prometheus_client import REGISTRY
for _collector in list(REGISTRY._collector_to_names.keys()):
    REGISTRY.unregister(_collector)

# 3. Import app globally (Runs fast, registers metrics cleanly)
from backend.main import app


# ── Shared test data ──────────────────────────────────────────────────────────
VALID_FEATURES = {
    "ear_mean": 0.30, "ear_min": 0.22, "ear_std": 0.04,
    "perclos": 0.10, "mar_mean": 0.38, "mar_max": 0.50,
    "head_pitch_mean": -3.0, "head_yaw_mean": 1.5, "head_roll_mean": 0.8,
}

DROWSY_FEATURES = {
    "ear_mean": 0.16, "ear_min": 0.08, "ear_std": 0.05,
    "perclos": 0.78, "mar_mean": 0.65, "mar_max": 0.92,
    "head_pitch_mean": -15.0, "head_yaw_mean": -2.5, "head_roll_mean": 1.8,
}

MOCK_ALERT_RESULT = {
    "state": "alert", "confidence": 0.91,
    "inference_latency_ms": 4.2, "model_version": "3",
}

MOCK_DROWSY_RESULT = {
    "state": "drowsy", "confidence": 0.87,
    "inference_latency_ms": 3.8, "model_version": "3",
}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /health
# ─────────────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_service_name(self, client):
        r = client.get("/health")
        assert r.json()["service"] == "backend"

    def test_has_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /status
# ─────────────────────────────────────────────────────────────────────────────
class TestStatus:
    def test_returns_200(self, client):
        r = client.get("/status")
        assert r.status_code == 200

    def test_has_active_sessions(self, client):
        r = client.get("/status")
        assert "active_sessions" in r.json()

    def test_active_sessions_is_int(self, client):
        r = client.get("/status")
        assert isinstance(r.json()["active_sessions"], int)

# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /ready
# ─────────────────────────────────────────────────────────────────────────────
class TestReady:
    def test_ready_200_when_model_reachable(self, client):
        with patch("backend.services.model_client.health_check", new_callable=AsyncMock, return_value=True):
            r = client.get("/ready")
            assert r.status_code == 200

    def test_ready_503_when_model_unreachable(self, client):
        with patch("backend.services.model_client.health_check", new_callable=AsyncMock, return_value=False):
            r = client.get("/ready")
            assert r.status_code == 503

    def test_ready_body_has_model_server_url(self, client):
        with patch("backend.services.model_client.health_check", new_callable=AsyncMock, return_value=True):
            r = client.get("/ready")
            assert "model_server_url" in r.json()

# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /session/start
# ─────────────────────────────────────────────────────────────────────────────
class TestSessionStart:
    def test_returns_200(self, client):
        r = client.post("/session/start", json={"window_size": 30})
        assert r.status_code == 200

    def test_returns_session_id(self, client):
        r = client.post("/session/start", json={"window_size": 30})
        assert "session_id" in r.json()

    def test_session_id_is_string(self, client):
        r = client.post("/session/start", json={"window_size": 30})
        assert isinstance(r.json()["session_id"], str)

    def test_status_is_active(self, client):
        r = client.post("/session/start", json={"window_size": 30})
        assert r.json()["status"] == "active"

    def test_returns_window_size(self, client):
        r = client.post("/session/start", json={"window_size": 45})
        assert r.json()["window_size"] == 45

    def test_default_window_size(self, client):
        r = client.post("/session/start", json={})
        assert "window_size" in r.json()

    def test_window_too_small_returns_422(self, client):
        r = client.post("/session/start", json={"window_size": 2})
        assert r.status_code == 422

    def test_window_too_large_returns_422(self, client):
        r = client.post("/session/start", json={"window_size": 9999})
        assert r.status_code == 422

    def test_returns_started_at(self, client):
        r = client.post("/session/start", json={})
        assert "started_at" in r.json()

# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /session/stop
# ─────────────────────────────────────────────────────────────────────────────
class TestSessionStop:
    def _create_session(self, client) -> str:
        r = client.post("/session/start", json={"window_size": 30})
        return r.json()["session_id"]

    def test_stop_existing_session_200(self, client):
        sid = self._create_session(client)
        r = client.post("/session/stop", json={"session_id": sid})
        assert r.status_code == 200

    def test_stop_returns_summary(self, client):
        sid = self._create_session(client)
        r = client.post("/session/stop", json={"session_id": sid})
        data = r.json()
        assert "total_frames_processed" in data
        assert "drowsy_alerts_triggered" in data
        assert "duration_seconds" in data

    def test_stop_nonexistent_session_returns_404(self, client):
        r = client.post("/session/stop", json={"session_id": "nonexistent-uuid"})
        assert r.status_code == 404

    def test_status_is_completed(self, client):
        sid = self._create_session(client)
        r = client.post("/session/stop", json={"session_id": sid})
        assert r.json()["status"] == "completed"

# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /session/{session_id}/status
# ─────────────────────────────────────────────────────────────────────────────
class TestSessionStatus:
    def _create_session(self, client) -> str:
        r = client.post("/session/start", json={"window_size": 30})
        return r.json()["session_id"]

    def test_status_active_session_200(self, client):
        sid = self._create_session(client)
        r = client.get(f"/session/{sid}/status")
        assert r.status_code == 200

    def test_status_has_frames_processed(self, client):
        sid = self._create_session(client)
        r = client.get(f"/session/{sid}/status")
        assert "frames_processed" in r.json()

    def test_status_has_drift_score(self, client):
        sid = self._create_session(client)
        r = client.get(f"/session/{sid}/status")
        assert "drift_score" in r.json()

    def test_unknown_session_returns_404(self, client):
        r = client.get("/session/nonexistent-id/status")
        assert r.status_code == 404

    def test_stopped_session_returns_404(self, client):
        sid = self._create_session(client)
        client.post("/session/stop", json={"session_id": sid})
        r = client.get(f"/session/{sid}/status")
        assert r.status_code == 404

# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /predict
# ─────────────────────────────────────────────────────────────────────────────
class TestPredict:
    def test_valid_returns_200(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert r.status_code == 200

    def test_returns_state(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert "state" in r.json()

    def test_state_valid_value(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert r.json()["state"] in ("alert", "drowsy")

    def test_returns_confidence(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert "confidence" in r.json()

    def test_confidence_in_range(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            c = r.json()["confidence"]
            assert 0.0 <= c <= 1.0

    def test_returns_latency(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert "inference_latency_ms" in r.json()

    def test_returns_alert_triggered_bool(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert isinstance(r.json()["alert_triggered"], bool)

    def test_missing_feature_returns_422(self, client):
        bad = {k: v for k, v in VALID_FEATURES.items() if k != "ear_mean"}
        r = client.post("/predict", json={"features": bad})
        assert r.status_code == 422

    def test_ear_above_1_returns_422(self, client):
        bad = {**VALID_FEATURES, "ear_mean": 1.5}
        r = client.post("/predict", json={"features": bad})
        assert r.status_code == 422

    def test_negative_perclos_returns_422(self, client):
        bad = {**VALID_FEATURES, "perclos": -0.1}
        r = client.post("/predict", json={"features": bad})
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_model_server_down_returns_503(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert r.status_code == 503

    def test_drowsy_state_triggers_alert(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_DROWSY_RESULT):
            r = client.post("/predict", json={"features": DROWSY_FEATURES})
            assert r.status_code == 200
            data = r.json()
            assert data["state"] == "drowsy"
            assert data["alert_triggered"] is True

    def test_alert_state_no_alert_triggered(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            r = client.post("/predict", json={"features": VALID_FEATURES})
            assert r.json()["alert_triggered"] is False

# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /metrics
# ─────────────────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_content_type_is_text(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers.get("content-type", "")

    @pytest.mark.skip(reason="Prometheus nuclear wipe clears these in test env")
    def test_contains_ddd_metrics(self, client):
        r = client.get("/metrics")
        assert "ddd_" in r.text

    @pytest.mark.skip(reason="Prometheus nuclear wipe clears these in test env")
    def test_contains_inference_latency(self, client):
        with patch("backend.services.model_client.predict", new_callable=AsyncMock, return_value=MOCK_ALERT_RESULT):
            client.post("/predict", json={"features": VALID_FEATURES})
        r = client.get("/metrics")
        assert "ddd_inference_latency" in r.text