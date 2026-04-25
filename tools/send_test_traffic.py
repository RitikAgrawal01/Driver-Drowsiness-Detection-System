"""
tools/send_test_traffic.py
───────────────────────────
Sends realistic mixed alert/drowsy traffic to the backend
to populate Grafana panels with real data before the demo.

Usage:
  # Warm up Grafana with 5 minutes of realistic traffic
  python tools/send_test_traffic.py --duration 300

  # Quick test (30 seconds)
  python tools/send_test_traffic.py --duration 30 --verbose
"""

import argparse
import random
import time

import requests

ALERT_FEATURES = {
    "ear_mean": 0.31, "ear_min": 0.23, "ear_std": 0.03,
    "perclos": 0.10,  "mar_mean": 0.37, "mar_max": 0.50,
    "head_pitch_mean": -2.5, "head_yaw_mean": 1.0, "head_roll_mean": 0.4,
}

DROWSY_FEATURES = {
    "ear_mean": 0.17, "ear_min": 0.09, "ear_std": 0.05,
    "perclos": 0.72,  "mar_mean": 0.58, "mar_max": 0.85,
    "head_pitch_mean": -12.0, "head_yaw_mean": -2.0, "head_roll_mean": 1.5,
}


def noisy(features: dict, scale: float = 0.02) -> dict:
    return {
        k: max(0.0, v + random.gauss(0, scale))
        for k, v in features.items()
    }


def run(backend_url: str, duration: int, verbose: bool):
    print(f"Sending test traffic for {duration}s to {backend_url}")

    try:
        requests.get(f"{backend_url}/health", timeout=3).raise_for_status()
    except Exception as e:
        print(f"Backend not reachable: {e}")
        return

    r = requests.post(f"{backend_url}/session/start",
                       json={"driver_id": "test_traffic", "window_size": 30})
    session_id = r.json().get("session_id")

    start = time.time()
    count = 0
    while time.time() - start < duration:
        # 80% alert, 20% drowsy (realistic distribution)
        features = (DROWSY_FEATURES if random.random() < 0.20
                    else ALERT_FEATURES)
        try:
            r = requests.post(
                f"{backend_url}/predict",
                json={"session_id": session_id, "features": noisy(features)},
                timeout=5,
            )
            count += 1
            if verbose and count % 20 == 0:
                data = r.json() if r.status_code == 200 else {}
                elapsed = time.time() - start
                print(f"  t={elapsed:.0f}s req={count} "
                      f"state={data.get('state','?')} "
                      f"conf={data.get('confidence', 0):.3f}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(0.1)  # ~10 req/sec

    requests.post(f"{backend_url}/session/stop",
                   json={"session_id": session_id})
    print(f"\n✓ Sent {count} requests in {duration}s. "
          f"Check Grafana at http://127.0.0.1:3001")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend-url", default="http://127.0.0.1:8000")
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run(args.backend_url, args.duration, args.verbose)
