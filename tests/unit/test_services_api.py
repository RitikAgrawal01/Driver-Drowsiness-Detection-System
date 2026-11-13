"""HTTP/WebSocket contracts of the backend and the model server (no real MLflow or MediaPipe)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ddd_core.schemas import FeatureVector
from ddd_core.windows import ALL_FEATURES

FEATURES = FeatureVector.model_config["json_schema_extra"]["example"]


def test_feature_contract_matches_window_features():
    assert list(FeatureVector.model_fields) == ALL_FEATURES


# ── model server ─────────────────────────────────────────────────────────
class FakePyfunc:
    def predict(self, df):
        s = df["perclos"].to_numpy()
        return pd.DataFrame({"score": s, "drowsy": (s >= 0.1).astype(int)})


@pytest.fixture
def model_server(monkeypatch):
    import model_server.main as ms
    from model_server.model_loader import store

    monkeypatch.setattr(ms, "try_load", lambda: False)  # no MLflow at startup
    monkeypatch.setattr(store, "_model", None)
    monkeypatch.setattr(store, "_meta", {})
    with TestClient(ms.app) as c:
        yield c, store


def load_fake(store, monkeypatch):
    monkeypatch.setattr(store, "_model", FakePyfunc())
    monkeypatch.setattr(
        store,
        "_meta",
        {"model_version": "3", "model_type": "rule", "threshold": 0.1, "window_s": 60, "features": ["perclos"]},
    )


def test_model_server_not_ready_returns_503(model_server):
    c, _ = model_server
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.post("/predict", json=FEATURES).status_code == 503


def test_model_server_predicts_with_model_threshold(model_server, monkeypatch):
    c, store = model_server
    load_fake(store, monkeypatch)
    r = c.post("/predict", json={**FEATURES, "perclos": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "drowsy" and body["threshold"] == 0.1 and body["window_s"] == 60
    assert c.post("/predict", json={**FEATURES, "perclos": 0.01}).json()["state"] == "alert"


def test_model_server_rejects_impossible_inputs(model_server, monkeypatch):
    c, store = model_server
    load_fake(store, monkeypatch)
    assert c.post("/predict", json={**FEATURES, "ear_mean": -0.1}).status_code == 422
    assert c.post("/predict", json={**FEATURES, "head_yaw_mean": 400}).status_code == 422
    missing = dict(FEATURES)
    missing.pop("blink_rate_pm")
    assert c.post("/predict", json=missing).status_code == 422


def test_model_server_inference_error_is_500_not_alert(model_server, monkeypatch):
    c, store = model_server
    load_fake(store, monkeypatch)

    class Broken:
        def predict(self, df):
            raise ValueError("bad model")

    monkeypatch.setattr(store, "_model", Broken())
    assert c.post("/predict", json=FEATURES).status_code == 500


def test_admin_reload_requires_token(model_server, monkeypatch):
    c, _ = model_server
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    assert c.post("/admin/reload").status_code == 401
    assert c.post("/admin/reload", headers={"X-Admin-Token": "secret"}).status_code == 500  # load fails: no MLflow


# ── backend ──────────────────────────────────────────────────────────────
MODEL_INFO = {
    "model_version": "3",
    "window_s": 15,
    "threshold": 0.1,
    "model_type": "rule",
    "target_fps": 10,
    "features": ["perclos"],
    "mlflow_run_id": None,
}


@pytest.fixture
def backend(monkeypatch):
    import backend.main as bm
    from backend.services import model_client
    from backend.services.serving import serving

    async def info():
        return dict(MODEL_INFO)

    async def predict(features):
        return {
            "state": "alert",
            "score": 0.01,
            "threshold": 0.1,
            "model_version": "3",
            "model_type": "rule",
            "window_s": 15,
            "inference_latency_ms": 0.1,
        }

    monkeypatch.setattr(model_client, "model_info", info)
    monkeypatch.setattr(model_client, "predict", predict)
    monkeypatch.setattr(serving, "_load_baseline", lambda run_id: None)
    serving.model = None
    with TestClient(bm.app) as c:
        yield c, model_client


def test_backend_health_ready_status(backend):
    c, _ = backend
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/ready").json()["window_s"] == 15
    assert c.get("/status").json()["model"]["model_version"] == "3"


def test_backend_ready_503_when_model_server_down(backend, monkeypatch):
    c, mc = backend
    from backend.services.serving import serving

    async def down():
        return None

    monkeypatch.setattr(mc, "model_info", down)
    serving.model = None
    assert c.get("/ready").status_code == 503


def test_backend_rest_predict_and_error_mapping(backend, monkeypatch):
    c, mc = backend
    assert c.post("/predict", json={"features": FEATURES}).json()["state"] == "alert"

    async def fail(features):
        raise mc.ModelServerError("connection_error", "down")

    monkeypatch.setattr(mc, "predict", fail)
    assert c.post("/predict", json={"features": FEATURES}).status_code == 503
    assert c.post("/predict", json={"features": {**FEATURES, "perclos": 2}}).status_code == 422


def test_cors_has_no_wildcard():
    from backend.config import Settings

    assert "*" not in Settings(cors_origins="http://a:3000, *").cors_origin_list


def test_websocket_protocol(backend, monkeypatch, jpeg_b64):
    c, _ = backend
    from backend.services.live_features import LiveFeatureExtractor

    def fake_process(self, img, t):
        f = {"face": 1, "ear": 0.3, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        self.add_frame(t, f)
        return f

    monkeypatch.setattr(LiveFeatureExtractor, "process_frame", fake_process)
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "frame", "ts": 0, "image_b64": jpeg_b64})
        assert ws.receive_json()["type"] == "error"  # init first
        ws.send_json({"type": "init", "window_size": 999})  # client value ignored
        ack = ws.receive_json()
        assert ack["type"] == "init_ack" and ack["window_s"] == 15 and ack["target_fps"] == 10
        last = None
        for i in range(170):
            ws.send_json({"type": "frame", "ts": i * 100, "image_b64": jpeg_b64})
            last = ws.receive_json()
        assert last["type"] == "status" and last["state"] == "alert"
        ws.send_json({"type": "close"})
        summary = ws.receive_json()
        assert summary["type"] == "session_summary" and summary["predictions"] >= 1
