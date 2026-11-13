"""Airflow helpers, the Alertmanager webhook and the single-source-of-truth checks."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import prometheus_webhook as webhook
import pytest
import yaml

import ddd_dag_utils as dag_utils
from tests.conftest import ROOT


# ── Airflow helpers ──────────────────────────────────────────────────────
def make_videos(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def test_validate_raw_videos_ok(tmp_path):
    make_videos(tmp_path, {"alert/alert1.mp4": b"a1", "drowsy/drowsy1.MOV": b"d1"})
    assert dag_utils.validate_raw_videos(tmp_path) == {"videos": 2, "participants": 1, "problems": []}


def test_validate_raw_videos_catches_duplicate_and_missing_pair(tmp_path):
    """v1 shipped drowsy8 as a byte-identical copy of alert8."""
    make_videos(tmp_path, {"alert/alert8.mp4": b"same", "drowsy/drowsy8.mp4": b"same", "alert/alert9.mp4": b"x"})
    with pytest.raises(ValueError) as e:
        dag_utils.validate_raw_videos(tmp_path)
    assert "identical content" in str(e.value) and "participant 9" in str(e.value)


@pytest.mark.parametrize(
    "conf,run_type,stale,expected",
    [
        ({"retrain_reason": "drift"}, "manual", [], True),
        ({}, "manual", [], True),
        ({}, "scheduled", ["train_evaluate"], True),
        ({}, "scheduled", [], False),
    ],
)
def test_should_retrain(conf, run_type, stale, expected):
    assert dag_utils.should_retrain(conf, run_type, stale)[0] is expected


def test_sh_fails_loudly(tmp_path):
    assert dag_utils.sh("echo hi", cwd=tmp_path).strip() == "hi"
    with pytest.raises(RuntimeError):
        dag_utils.sh("false", cwd=tmp_path)


def test_no_command_swallows_failures():
    """v1 appended `|| true` to commands in the DAGs and CI, so failures were invisible."""
    import re

    pattern = re.compile(r"""(sh\(|run:|bash -c)[^\n]*\|\|\s*true""")
    files = list((ROOT / "airflow" / "dags").glob("*.py")) + [ROOT / ".github" / "workflows" / "ci.yml"]
    for f in files:
        assert not pattern.search(f.read_text()), f


# ── webhook ──────────────────────────────────────────────────────────────
def alert(name, action, status="firing"):
    return {"status": status, "labels": {"alertname": name, "action": action}}


def test_select_retrain_alerts_uses_labels_not_names():
    payload = {
        "alerts": [
            alert("DataDriftDetected", "retrain"),
            alert("HighPredictionErrorRate", "page"),
            alert("DataDriftDetected", "retrain", status="resolved"),
        ]
    }
    assert webhook.select_retrain_alerts(payload) == ["DataDriftDetected"]


def test_cooldown():
    assert webhook.cooldown_ok(0, 7 * 3600, 6) and not webhook.cooldown_ok(0, 3600, 6)


@pytest.fixture
def server(monkeypatch):
    calls = []
    monkeypatch.setattr(webhook, "trigger_airflow", lambda reason, conf: (calls.append(reason) or True, "ok"))
    monkeypatch.setitem(webhook.STATE, "last_trigger", 0.0)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), webhook.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()


def post(url, body, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_webhook_triggers_once_then_cooldown(server):
    url, calls = server
    payload = {"alerts": [alert("DataDriftDetected", "retrain")]}
    assert post(f"{url}/webhook", payload) == (200, {"triggered": True, "detail": "ok"})
    assert post(f"{url}/webhook", payload)[1]["reason"] == "cooldown"
    assert calls == ["DataDriftDetected"]


def test_webhook_pages_are_logged_not_retrained(server):
    url, calls = server
    status, body = post(f"{url}/notify", {"alerts": [alert("ModelServerDown", "page")]})
    assert status == 200 and body == {"logged": 1} and calls == []
    assert post(f"{url}/webhook", {"alerts": [alert("ModelServerDown", "page")]})[1]["triggered"] is False


def test_manual_trigger_requires_token(server, monkeypatch):
    url, _ = server
    monkeypatch.setenv("WEBHOOK_TOKEN", "t0k")
    assert post(f"{url}/trigger", {"reason": "demo"})[0] == 401
    assert post(f"{url}/trigger", {"reason": "demo"}, {"X-Webhook-Token": "t0k"})[0] == 200


# ── one source of truth ──────────────────────────────────────────────────
def test_alert_rules_and_dashboard_generated_from_params():
    import render_alert_rules as r  # noqa: E402  (scripts/ on sys.path below)

    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    assert (ROOT / "monitoring/prometheus/alert_rules.yml").read_text() == r.render(params)
    assert (ROOT / "monitoring/grafana/dashboards/ddd_main.json").read_text() == r.render_dashboard(params)


def test_dvc_lock_matches_params():
    import check_params_lock

    assert check_params_lock.check() == []


def test_serving_fps_equals_training_fps(params):
    assert params["serving"]["target_fps"] == params["extract"]["target_fps"]
    assert params["serving"]["max_side"] == params["extract"]["max_side"]


def test_env_example_has_no_model_settings():
    text = (ROOT / ".env.example").read_text()
    for key in ["SLIDING_WINDOW_SIZE", "FRAME_RATE", "EAR_THRESHOLD", "CONFIDENCE_THRESHOLD", "DRIFT_SCORE"]:
        assert key not in text
