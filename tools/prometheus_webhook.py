"""
tools/prometheus_webhook.py — Alertmanager -> Airflow bridge (runs as the `webhook` compose service).

  POST /webhook  Alertmanager "retrain" receiver: firing alerts labelled action=retrain trigger
                 the ddd_retrain_pipeline DAG (once per cooldown window, params.yaml retrain.cooldown_hours)
  POST /notify   Alertmanager "page" receiver: operational alerts are logged (never retrain)
  POST /trigger  manual trigger for demos (requires X-Webhook-Token when WEBHOOK_TOKEN is set)
  GET  /health, /metrics

v1 fixes: Prometheus now talks to Alertmanager (v1 posted Alertmanager-format alerts straight to
this server on a path it did not serve); retraining is selected by label, not a hard-coded list
that included HighErrorRate; credentials come only from the environment (v1 defaulted to admin/admin).
"""

import base64
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] webhook — %(message)s")
log = logging.getLogger("webhook")

DAG_ID = "ddd_retrain_pipeline"
STATE = {
    "last_trigger": 0.0,
    "counts": {"received": 0, "triggered": 0, "suppressed_cooldown": 0, "trigger_failed": 0, "notified": 0},
}


def cooldown_hours() -> float:
    path = os.getenv("DDD_PARAMS", "params.yaml")
    with open(path) as f:
        return float(yaml.safe_load(f)["retrain"]["cooldown_hours"])


def select_retrain_alerts(payload: dict) -> list[str]:
    """Names of FIRING alerts that ask for retraining (label action=retrain)."""
    return sorted(
        {
            a.get("labels", {}).get("alertname", "unknown")
            for a in payload.get("alerts", [])
            if a.get("status") == "firing" and a.get("labels", {}).get("action") == "retrain"
        }
    )


def cooldown_ok(last: float, now: float, hours: float) -> bool:
    return now - last >= hours * 3600


def trigger_airflow(reason: str, conf: dict) -> tuple[bool, str]:
    url, user, pwd = os.getenv("AIRFLOW_URL"), os.getenv("AIRFLOW_USER"), os.getenv("AIRFLOW_PASSWORD")
    if not (url and user and pwd):
        return False, "AIRFLOW_URL / AIRFLOW_USER / AIRFLOW_PASSWORD not set"
    run_id = f"webhook__{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    body = json.dumps({"dag_run_id": run_id, "conf": {"retrain_reason": reason, **conf}}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/dags/{DAG_ID}/dagRuns",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode(),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return True, f"{r.status} {run_id}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:
        return False, str(e)


def metrics_text() -> str:
    lines = []
    for k, v in STATE["counts"].items():
        lines += [f"# TYPE ddd_webhook_{k}_total counter", f"ddd_webhook_{k}_total {v}"]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/metrics":
            return self._send(200, metrics_text(), "text/plain; version=0.0.4")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        STATE["counts"]["received"] += 1

        if self.path == "/notify":
            STATE["counts"]["notified"] += 1
            for a in payload.get("alerts", []):
                log.warning(
                    "PAGE %s [%s] %s",
                    a.get("labels", {}).get("alertname"),
                    a.get("status"),
                    a.get("annotations", {}).get("summary", ""),
                )
            return self._send(200, {"logged": len(payload.get("alerts", []))})

        if self.path == "/trigger":
            expected = os.getenv("WEBHOOK_TOKEN")
            token = self.headers.get("X-Webhook-Token")
            if expected and not (token and hmac.compare_digest(token, expected)):
                return self._send(401, {"error": "invalid token"})
            names = [payload.get("reason", "manual_trigger")]
        elif self.path == "/webhook":
            names = select_retrain_alerts(payload)
            if not names:
                return self._send(200, {"triggered": False, "reason": "no firing action=retrain alerts"})
        else:
            return self._send(404, {"error": f"unknown path {self.path}"})

        now = time.time()
        if self.path == "/webhook" and not cooldown_ok(STATE["last_trigger"], now, cooldown_hours()):
            STATE["counts"]["suppressed_cooldown"] += 1
            return self._send(200, {"triggered": False, "reason": "cooldown"})
        ok, detail = trigger_airflow(";".join(names), {"alerts": names})
        if ok:
            STATE["last_trigger"] = now
            STATE["counts"]["triggered"] += 1
            log.info("triggered %s: %s (%s)", DAG_ID, names, detail)
            return self._send(200, {"triggered": True, "detail": detail})
        STATE["counts"]["trigger_failed"] += 1
        log.error("trigger failed: %s", detail)
        return self._send(502, {"triggered": False, "error": detail})  # Alertmanager will retry


def main():
    port = int(os.getenv("WEBHOOK_PORT", "9095"))
    log.info("listening on :%d (DAG %s)", port, DAG_ID)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
