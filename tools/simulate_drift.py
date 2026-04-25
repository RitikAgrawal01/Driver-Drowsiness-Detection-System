"""
tools/simulate_drift.py
────────────────────────
Drift simulation tool for the DDD system demo.

Sends POST requests to the backend /predict endpoint with feature vectors
that are systematically shifted from the training baseline, simulating
real-world distribution shift (lighting change, camera angle, sunglasses).

When enough drifted requests accumulate, the backend's drift_detector.py
will compute a high KL-divergence score, the Prometheus gauge
ddd_overall_drift_score will exceed 0.15, and the DataDriftDetected
alert will fire in Grafana.

Usage:
  # Normal traffic (no drift) — establish baseline reading
  python tools/simulate_drift.py --mode normal --count 200

  # Simulate lighting drift (EAR drops, PERCLOS rises)
  python tools/simulate_drift.py --mode lighting --count 300

  # Simulate camera angle drift (head pose shifts)
  python tools/simulate_drift.py --mode angle --count 300

  # Simulate sunglasses (EAR artificially low)
  python tools/simulate_drift.py --mode sunglasses --count 300

  # Watch drift score in real time (requires requests + rich)
  python tools/simulate_drift.py --mode lighting --count 500 --verbose

  # Custom shift amounts
  python tools/simulate_drift.py --mode custom \\
      --shift-ear -0.10 --shift-perclos 0.30 --count 300
"""

import argparse
import json
import math
import random
import sys
import time
from typing import Optional

import requests

# ── Baseline feature distributions (from training) ───────────────────────────
# These match what the model was trained on (normal alert/drowsy driver)
BASELINE_ALERT = {
    "ear_mean":        0.30,
    "ear_min":         0.22,
    "ear_std":         0.04,
    "perclos":         0.12,
    "mar_mean":        0.38,
    "mar_max":         0.52,
    "head_pitch_mean": -3.0,
    "head_yaw_mean":    1.2,
    "head_roll_mean":   0.5,
}

# ── Drift scenario definitions ────────────────────────────────────────────────
DRIFT_SCENARIOS = {
    "normal": {
        "description": "Normal alert driver — no drift",
        "shifts": {},
        "noise_scale": 0.02,
    },
    "lighting": {
        "description": "Dark lighting — MediaPipe sees eyes as more closed",
        "shifts": {
            "ear_mean":   -0.08,   # eyes appear more closed
            "ear_min":    -0.06,
            "perclos":    +0.25,   # more frames counted as closed
            "mar_mean":   +0.05,
        },
        "noise_scale": 0.03,
    },
    "angle": {
        "description": "Camera angle shift — head pose distribution moves",
        "shifts": {
            "head_pitch_mean": -8.0,   # camera now above driver
            "head_yaw_mean":   +6.0,   # driver looking slightly right
            "head_roll_mean":  +3.0,
            "ear_mean":        -0.03,  # slight EAR change due to perspective
        },
        "noise_scale": 0.04,
    },
    "sunglasses": {
        "description": "Driver wearing sunglasses — EAR near zero",
        "shifts": {
            "ear_mean":   -0.20,   # sunglasses block eye visibility
            "ear_min":    -0.18,
            "ear_std":    -0.02,
            "perclos":    +0.60,   # nearly all frames counted as closed
        },
        "noise_scale": 0.01,
    },
    "combined": {
        "description": "Combined: dark + angle + fatigue",
        "shifts": {
            "ear_mean":        -0.12,
            "ear_min":         -0.10,
            "perclos":         +0.35,
            "head_pitch_mean": -6.0,
            "mar_mean":        +0.08,
            "mar_max":         +0.12,
        },
        "noise_scale": 0.04,
    },
}


def generate_feature_vector(
    base: dict,
    shifts: dict,
    noise_scale: float,
    gradual: bool = True,
    step: int = 0,
    total_steps: int = 300,
) -> dict:
    """
    Generate a single shifted feature vector.

    Args:
        base:         baseline feature values
        shifts:       dict of {feature: shift_amount}
        noise_scale:  gaussian noise std added to each feature
        gradual:      if True, shift increases linearly with step
        step:         current request number (for gradual ramp)
        total_steps:  total requests to ramp over

    Returns:
        dict of feature values ready to POST to /predict
    """
    progress = min(1.0, step / max(total_steps * 0.6, 1)) if gradual else 1.0
    vector = {}

    for key, base_val in base.items():
        shift = shifts.get(key, 0.0) * progress
        noise = random.gauss(0, noise_scale)
        value = base_val + shift + noise

        # Clamp to physically valid ranges
        if key in ("ear_mean", "ear_min", "perclos"):
            value = max(0.0, min(1.0, value))
        elif key in ("ear_std", "mar_mean", "mar_max"):
            value = max(0.0, value)

        vector[key] = round(value, 6)

    return vector


def run_simulation(
    backend_url: str,
    mode: str,
    count: int,
    delay_sec: float,
    verbose: bool,
    custom_shifts: Optional[dict] = None,
    gradual: bool = True,
):
    """
    Send `count` requests to /predict with shifted features.
    Prints drift score from response headers / status as it progresses.
    """
    if mode == "custom" and custom_shifts:
        scenario = {
            "description": f"Custom shifts: {custom_shifts}",
            "shifts": custom_shifts,
            "noise_scale": 0.03,
        }
    elif mode in DRIFT_SCENARIOS:
        scenario = DRIFT_SCENARIOS[mode]
    else:
        print(f"Unknown mode: {mode}. Choose from: {list(DRIFT_SCENARIOS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DDD Drift Simulation")
    print(f"  Mode        : {mode}")
    print(f"  Description : {scenario['description']}")
    print(f"  Requests    : {count}")
    print(f"  Backend URL : {backend_url}")
    print(f"  Gradual     : {gradual}")
    print(f"{'='*60}\n")

    # First check backend is up
    try:
        r = requests.get(f"{backend_url}/health", timeout=5)
        r.raise_for_status()
        print("✓ Backend is reachable\n")
    except Exception as e:
        print(f"✗ Cannot reach backend at {backend_url}: {e}")
        print("  Make sure the backend is running: uvicorn backend.main:app --port 8000")
        sys.exit(1)

    # Start a simulation session
    try:
        r = requests.post(
            f"{backend_url}/session/start",
            json={"driver_id": f"drift_sim_{mode}", "window_size": 30},
            timeout=5,
        )
        session_id = r.json().get("session_id", "unknown")
        print(f"✓ Session started: {session_id}\n")
    except Exception as e:
        print(f"Warning: Could not start session: {e}")
        session_id = "unknown"

    success_count = 0
    error_count = 0
    alert_count = 0
    last_state = "alert"
    start_time = time.time()

    for i in range(count):
        features = generate_feature_vector(
            base=BASELINE_ALERT,
            shifts=scenario["shifts"],
            noise_scale=scenario["noise_scale"],
            gradual=gradual,
            step=i,
            total_steps=count,
        )

        try:
            r = requests.post(
                f"{backend_url}/predict",
                json={"session_id": session_id, "features": features},
                timeout=10,
            )

            if r.status_code == 200:
                data = r.json()
                last_state = data.get("state", "?")
                confidence = data.get("confidence", 0)
                alert = data.get("alert_triggered", False)
                success_count += 1
                if alert:
                    alert_count += 1

                if verbose or (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    progress_pct = round((i + 1) / count * 100)
                    rps = round(success_count / elapsed, 1)
                    print(
                        f"  [{progress_pct:3d}%] req={i+1:4d} "
                        f"state={last_state:7s} "
                        f"conf={confidence:.3f} "
                        f"alert={'YES' if alert else 'no ':4s} "
                        f"rps={rps} "
                        f"ear_mean={features['ear_mean']:.3f} "
                        f"perclos={features['perclos']:.3f}"
                    )
            else:
                error_count += 1
                if verbose:
                    print(f"  [ERROR] req={i+1} status={r.status_code}: {r.text[:100]}")

        except requests.exceptions.ConnectionError:
            error_count += 1
            print(f"  [ERROR] Connection refused at req={i+1}")
            break
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"  [ERROR] req={i+1}: {e}")

        if delay_sec > 0:
            time.sleep(delay_sec)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"  Requests sent    : {count}")
    print(f"  Successful       : {success_count}")
    print(f"  Errors           : {error_count}")
    print(f"  Drowsy alerts    : {alert_count}")
    print(f"  Elapsed          : {elapsed:.1f}s")
    print(f"  Throughput       : {round(success_count/elapsed, 1)} req/s")
    print(f"{'='*60}")
    print()
    print("Now check Grafana at http://127.0.0.1:3001")
    print("Look for: 'Feature Drift Scores' panel rising above 0.15")
    print("And:      'DataDriftDetected' alert firing in Alerting tab")
    print()

    # Stop session
    if session_id != "unknown":
        try:
            requests.post(
                f"{backend_url}/session/stop",
                json={"session_id": session_id},
                timeout=5,
            )
            print(f"✓ Session {session_id} stopped")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate data drift for DDD system demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        default="lighting",
        choices=list(DRIFT_SCENARIOS.keys()) + ["custom"],
        help="Drift scenario to simulate (default: lighting)",
    )
    parser.add_argument(
        "--count", type=int, default=300,
        help="Number of requests to send (default: 300)",
    )
    parser.add_argument(
        "--backend-url", default="http://127.0.0.1:8000",
        help="Backend API URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Delay between requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every request result",
    )
    parser.add_argument(
        "--no-gradual", action="store_true",
        help="Apply full shift immediately (default: gradual ramp)",
    )
    # Custom shift overrides
    parser.add_argument("--shift-ear",     type=float, default=0.0)
    parser.add_argument("--shift-perclos", type=float, default=0.0)
    parser.add_argument("--shift-pitch",   type=float, default=0.0)
    parser.add_argument("--shift-mar",     type=float, default=0.0)

    args = parser.parse_args()

    custom_shifts = None
    if args.mode == "custom":
        custom_shifts = {}
        if args.shift_ear:
            custom_shifts["ear_mean"] = args.shift_ear
            custom_shifts["ear_min"] = args.shift_ear * 0.8
        if args.shift_perclos:
            custom_shifts["perclos"] = args.shift_perclos
        if args.shift_pitch:
            custom_shifts["head_pitch_mean"] = args.shift_pitch
        if args.shift_mar:
            custom_shifts["mar_mean"] = args.shift_mar
            custom_shifts["mar_max"] = args.shift_mar * 1.3

    run_simulation(
        backend_url=args.backend_url,
        mode=args.mode,
        count=args.count,
        delay_sec=args.delay,
        verbose=args.verbose,
        custom_shifts=custom_shifts,
        gradual=not args.no_gradual,
    )
