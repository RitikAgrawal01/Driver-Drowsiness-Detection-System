"""Shared test setup: real params.yaml, no network, importable DAG utils and tools."""

import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "airflow" / "dags"), str(ROOT / "tools"), str(ROOT / "scripts")]
os.environ.setdefault("DDD_PARAMS", str(ROOT / "params.yaml"))
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:9"  # unreachable on purpose: tests never hit a server
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")


@pytest.fixture
def params():
    from ddd_core.config import load_params

    return load_params()


@pytest.fixture
def jpeg_b64():
    img = np.full((48, 64, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def eye_points(ear: float, width: float = 30.0, x0: float = 100.0, y0: float = 100.0) -> list:
    """Six eye landmarks (p1..p6) whose EAR is exactly `ear`."""
    h = ear * width  # each vertical distance = ear * width  ->  EAR = 2h / (2w) = ear
    return [
        (x0, y0),
        (x0 + width / 3, y0 - h / 2),
        (x0 + 2 * width / 3, y0 - h / 2),
        (x0 + width, y0),
        (x0 + 2 * width / 3, y0 + h / 2),
        (x0 + width / 3, y0 + h / 2),
    ]


def face_points(ear: float = 0.3, mouth_open: float = 0.0) -> dict:
    """Pixel landmarks for every index ddd_core uses: an upright face looking at the camera."""
    from ddd_core.geometry import FACE_3D, LEFT_EYE, POSE, RIGHT_EYE

    pts = {}
    for idx, p in zip(RIGHT_EYE, eye_points(ear, x0=250)):
        pts[idx] = p
    for idx, p in zip(LEFT_EYE, eye_points(ear, x0=360)):
        pts[idx] = p
    pts[61], pts[291] = (290.0, 330.0), (350.0, 330.0)
    pts[13], pts[14] = (320.0, 330.0 - 30 * mouth_open), (320.0, 330.0 + 30 * mouth_open)
    # head-pose points: project the 3D model with no rotation, 600 mm in front of the camera
    w, h, f = 640, 480, 640.0
    for idx, (X, Y, Z) in zip(POSE, FACE_3D):
        pts[idx] = (f * X / (Z + 600) + w / 2, f * Y / (Z + 600) + h / 2)
    return pts
