"""
tools/prometheus_webhook.py
────────────────────────────
Lightweight webhook server that receives Prometheus Alertmanager
notifications and triggers the Airflow retraining DAG.

This bridges the gap between Prometheus firing a DataDriftDetected
alert and Airflow automatically kicking off the retraining pipeline.

Architecture:
  Prometheus → alert fires → Alertmanager → POST /webhook
  → this server → POST Airflow /api/v1/dags/ddd_retrain_pipeline/dagRuns

Usage:
  python tools/prometheus_webhook.py          # starts on port 9095
  python tools/prometheus_webhook.py --port 9095

For demo purposes, you can also trigger manually:
  curl -X POST http://127.0.0.1:9095/trigger \
    -H "Content-Type: application/json" \
    -d '{"reason": "manual_demo"}'
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
)
logger = logging.getLogger("prometheus_webhook")

AIRFLOW_URL     = os.getenv("AIRFLOW_URL",     "http://127.0.0.1:8080")
AIRFLOW_USER    = os.getenv("AIRFLOW_USER",    "admin")
AIRFLOW_PASS    = os.getenv("AIRFLOW_PASS",    "admin")
AIRFLOW_DAG_ID  = "ddd_retrain_pipeline"

# Alerts that should trigger retraining
TRIGGER_ALERTS = {
    "DataDriftDetected",
    "HighErrorRate",
    "EARFeatureDrift",
    "PERCLOSFeatureDrift",
}


def trigger_airflow_dag(reason: str, conf: dict = None) -> bool:
    """
    POST to Airflow REST API to trigger the retraining DAG.

    Args:
        reason: human-readable reason for the trigger
        conf:   optional DAG run config dict

    Returns:
        True if DAG was triggered successfully
    """
    dag_run_id = f"webhook__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    payload = {
        "dag_run_id": dag_run_id,
        "conf": conf or {"retrain_reason": reason},
        "note": f"Triggered by Prometheus webhook: {reason}",
    }

    try:
        resp = requests.post(
            f"{AIRFLOW_URL}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns",
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(
                f"✓ Airflow DAG triggered successfully\n"
                f"  DAG ID   : {AIRFLOW_DAG_ID}\n"
                f"  Run ID   : {data.get('dag_run_id', dag_run_id)}\n"
                f"  Reason   : {reason}\n"
                f"  State    : {data.get('state', 'queued')}"
            )
            return True
        elif resp.status_code == 409:
            logger.warning(
                "DAG run already exists or DAG is paused. "
                "Check Airflow UI to ensure the DAG is unpaused."
            )
            return False
        else:
            logger.error(
                f"Failed to trigger DAG: HTTP {resp.status_code}\n"
                f"  Response: {resp.text[:200]}"
            )
            return False

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Cannot connect to Airflow at {AIRFLOW_URL}. "
            "Is Airflow running? Start with: make airflow-up"
        )
        return False
    except Exception as e:
        logger.error(f"DAG trigger exception: {e}")
        return False


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Prometheus Alertmanager webhooks."""

    def log_message(self, format, *args):
        """Suppress default access log — we use structured logging instead."""
        pass

    def _send_response(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        """Handle POST requests from Prometheus Alertmanager or manual triggers."""

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON"})
            return

        # ── Manual trigger endpoint ───────────────────────────────────────────
        if self.path == "/trigger":
            reason = data.get("reason", "manual_trigger")
            conf   = data.get("conf", {})
            logger.info(f"Manual trigger received: {reason}")
            success = trigger_airflow_dag(reason, conf)
            self._send_response(
                200 if success else 500,
                {"triggered": success, "reason": reason}
            )
            return

        # ── Prometheus Alertmanager webhook ───────────────────────────────────
        if self.path == "/webhook":
            alerts = data.get("alerts", [])
            if not alerts:
                self._send_response(200, {"status": "no_alerts"})
                return

            triggered = False
            for alert in alerts:
                alert_name = alert.get("labels", {}).get("alertname", "")
                status     = alert.get("status", "")  # "firing" or "resolved"

                if status != "firing":
                    continue

                if alert_name in TRIGGER_ALERTS:
                    logger.info(f"Prometheus alert firing: {alert_name}")
                    reason = f"prometheus_alert:{alert_name}"
                    conf = {
                        "retrain_reason": reason,
                        "alert_name":     alert_name,
                        "alert_labels":   alert.get("labels", {}),
                    }
                    if trigger_airflow_dag(reason, conf):
                        triggered = True
                        break  # only trigger once per webhook call
                else:
                    logger.debug(f"Alert {alert_name} does not trigger retraining")

            self._send_response(
                200,
                {"received": len(alerts), "triggered": triggered}
            )
            return

        # ── Health check ──────────────────────────────────────────────────────
        if self.path == "/health":
            self._send_response(200, {"status": "ok", "service": "prometheus_webhook"})
            return

        self._send_response(404, {"error": f"Unknown path: {self.path}"})

    def do_GET(self):
        if self.path == "/health":
            self._send_response(200, {"status": "ok"})
        else:
            self._send_response(404, {"error": "Not found"})


def run(port: int = 9095):
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    logger.info(
        f"\n{'='*55}\n"
        f"Prometheus → Airflow Webhook Server\n"
        f"  Listening  : http://0.0.0.0:{port}\n"
        f"  Airflow    : {AIRFLOW_URL}\n"
        f"  DAG        : {AIRFLOW_DAG_ID}\n"
        f"  Triggers on: {', '.join(sorted(TRIGGER_ALERTS))}\n"
        f"\n"
        f"  Manual trigger:\n"
        f"    curl -X POST http://127.0.0.1:{port}/trigger\n"
        f"      -H 'Content-Type: application/json'\n"
        f"      -d '{{\"reason\": \"demo\"}}'\n"
        f"{'='*55}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Webhook server stopped")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prometheus → Airflow webhook bridge"
    )
    parser.add_argument("--port", type=int, default=9095)
    parser.add_argument("--airflow-url", default=AIRFLOW_URL)
    parser.add_argument("--airflow-user", default=AIRFLOW_USER)
    parser.add_argument("--airflow-pass", default=AIRFLOW_PASS)
    args = parser.parse_args()

    AIRFLOW_URL  = args.airflow_url
    AIRFLOW_USER = args.airflow_user
    AIRFLOW_PASS = args.airflow_pass

    run(args.port)
