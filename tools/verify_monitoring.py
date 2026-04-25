"""
tools/verify_monitoring.py
───────────────────────────
Verifies all monitoring stack components are running and correctly wired.
Run this after docker-compose up to confirm Phase 6 is working.

Usage:
  python tools/verify_monitoring.py

Expected output: all checks GREEN before proceeding to demo.
"""

import sys
import time
import requests

CHECKS = [
    ("Backend /health",      "http://127.0.0.1:8000/health",   200),
    ("Backend /metrics",     "http://127.0.0.1:8000/metrics",  200),
    ("Backend /ready",       "http://127.0.0.1:8000/ready",    None),  # 200 or 503
    ("Model Server /health", "http://127.0.0.1:8001/health",   200),
    ("Model Server /metrics","http://127.0.0.1:8001/metrics",  200),
    ("Prometheus UI",        "http://127.0.0.1:9090/-/ready",  200),
    ("Grafana UI",           "http://127.0.0.1:3001/api/health", 200),
    ("MLflow UI",            "http://127.0.0.1:5000/health",   200),
]

PROMETHEUS_METRICS = [
    "ddd_inference_latency_seconds",
    "ddd_prediction_confidence",
    "ddd_feature_drift_score",
    "ddd_overall_drift_score",
    "ddd_requests_total",
    "ddd_active_sessions",
    "ddd_drowsy_alerts_total",
    "ddd_frames_processed_total",
    "ddd_model_server_reachable",
]


def check_url(name: str, url: str, expected_status: int) -> bool:
    try:
        r = requests.get(url, timeout=5)
        ok = (expected_status is None) or (r.status_code == expected_status)
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name:<30} {r.status_code} {'OK' if ok else 'FAILED'}")
        return ok
    except Exception as e:
        print(f"  ✗ {name:<30} UNREACHABLE — {e}")
        return False


def check_prometheus_metrics() -> bool:
    print("\n── Prometheus metric presence ──────────────────────────────")
    try:
        r = requests.get("http://127.0.0.1:8000/metrics", timeout=5)
        body = r.text
        all_ok = True
        for metric in PROMETHEUS_METRICS:
            present = metric in body
            symbol = "✓" if present else "✗"
            print(f"  {symbol} {metric}")
            if not present:
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  ✗ Could not fetch /metrics: {e}")
        return False


def check_prometheus_targets() -> bool:
    print("\n── Prometheus scrape targets ───────────────────────────────")
    try:
        r = requests.get(
            "http://127.0.0.1:9090/api/v1/targets", timeout=5
        )
        if r.status_code != 200:
            print(f"  ✗ Prometheus API returned {r.status_code}")
            return False
        data = r.json()
        targets = data.get("data", {}).get("activeTargets", [])
        all_up = True
        for t in targets:
            job = t.get("labels", {}).get("job", "?")
            health = t.get("health", "?")
            symbol = "✓" if health == "up" else "✗"
            print(f"  {symbol} {job:<25} {health}")
            if health != "up":
                all_up = False
        return all_up
    except Exception as e:
        print(f"  ✗ Could not reach Prometheus API: {e}")
        return False


def check_alert_rules() -> bool:
    print("\n── Prometheus alert rules ──────────────────────────────────")
    try:
        r = requests.get(
            "http://127.0.0.1:9090/api/v1/rules", timeout=5
        )
        data = r.json()
        groups = data.get("data", {}).get("groups", [])
        rule_names = []
        for g in groups:
            for rule in g.get("rules", []):
                rule_names.append(rule.get("name", "?"))
        expected = [
            "HighErrorRate", "HighInferenceLatency",
            "DataDriftDetected", "ModelServerDown",
        ]
        all_ok = True
        for name in expected:
            present = name in rule_names
            symbol = "✓" if present else "✗"
            print(f"  {symbol} Alert rule: {name}")
            if not present:
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  ✗ Could not check alert rules: {e}")
        return False


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   DDD Monitoring Stack Verification                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    print("── Service health checks ───────────────────────────────────")
    results = [check_url(name, url, status) for name, url, status in CHECKS]

    metrics_ok = check_prometheus_metrics()
    targets_ok = check_prometheus_targets()
    rules_ok = check_alert_rules()

    all_passed = all(results) and metrics_ok and rules_ok

    print("\n╔══════════════════════════════════════════════════════════╗")
    if all_passed:
        print("║   ALL CHECKS PASSED ✓                                   ║")
        print("║                                                          ║")
        print("║   Next steps:                                            ║")
        print("║   1. Open Grafana: http://127.0.0.1:3001 (admin/admin)   ║")
        print("║   2. Warm up metrics: python tools/send_test_traffic.py  ║")
        print("║   3. Simulate drift:  python tools/simulate_drift.py     ║")
        print("║              --mode lighting --count 300                 ║")
    else:
        print("║   SOME CHECKS FAILED ✗  — see above for details         ║")
        print("║                                                          ║")
        print("║   Common fixes:                                          ║")
        print("║   • Run: docker-compose up -d                            ║")
        print("║   • Check: docker-compose ps                             ║")
        print("║   • Logs: docker-compose logs backend                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    sys.exit(0 if all_passed else 1)
