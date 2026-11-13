"""Per-frame geometry from MediaPipe FaceMesh landmarks (inputs in PIXEL coordinates)."""

import math

import cv2
import numpy as np

# Eye order: p1 outer corner, p2, p3 upper lid, p4 inner corner, p5, p6 lower lid
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth corners 61/291 and INNER lips 13/14 (v1 used outer lips 0/17, which never close).
MOUTH = [61, 291, 13, 14]
# Head pose: nose tip, chin, right-eye outer, left-eye outer, mouth corners
POSE = [1, 152, 226, 446, 57, 287]
USED_LANDMARKS = sorted(set(RIGHT_EYE + LEFT_EYE + MOUTH + POSE))

# Generic 3D face in OpenCV camera convention (x right, y DOWN, z away from camera).
# v1 used y-up model points against y-down image points (a reflection), which gave
# pitch near +/-180 degrees for an upright face.
FACE_3D = np.array([
    [0.0, 0.0, 0.0],        # nose tip (1)
    [0.0, 63.6, 12.5],      # chin (152)
    [-43.3, -32.7, 26.0],   # right-eye outer corner (226), image left
    [43.3, -32.7, 26.0],    # left-eye outer corner (446), image right
    [-28.9, 28.9, 24.1],    # mouth corner (57)
    [28.9, 28.9, 24.1],     # mouth corner (287)
], dtype=np.float64)

NAN = float("nan")


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts: dict, idx: list) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|); ~0.3 open, -> 0 closed."""
    p = [pts[i] for i in idx]
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return NAN
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(pts: dict) -> float:
    """Inner-lip opening / mouth width; ~0 closed."""
    width = _dist(pts[61], pts[291])
    if width < 1e-6:
        return NAN
    return _dist(pts[13], pts[14]) / width


def head_pose(pts: dict, w: int, h: int) -> tuple:
    """(pitch, yaw, roll) in degrees; (0, 0, 0) = upright, facing the camera."""
    img = np.array([pts[i] for i in POSE], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, img, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return NAN, NAN, NAN
    r, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(r[0, 0], r[1, 0])
    return (math.degrees(math.atan2(r[2, 1], r[2, 2])),
            math.degrees(math.atan2(-r[2, 0], sy)),
            math.degrees(math.atan2(r[1, 0], r[0, 0])))


def frame_features(pts: dict | None, w: int, h: int) -> dict:
    """All per-frame features. pts=None (no face) -> face=0 and NaNs."""
    if pts is None:
        return {"face": 0, "ear": NAN, "mar": NAN, "pitch": NAN, "yaw": NAN, "roll": NAN}
    pitch, yaw, roll = head_pose(pts, w, h)
    return {
        "face": 1,
        "ear": (eye_aspect_ratio(pts, RIGHT_EYE) + eye_aspect_ratio(pts, LEFT_EYE)) / 2.0,
        "mar": mouth_aspect_ratio(pts),
        "pitch": pitch, "yaw": yaw, "roll": roll,
    }


def landmarks_to_pixels(landmarks, w: int, h: int) -> dict:
    """MediaPipe normalized landmarks -> {index: (x_px, y_px)} for the indices we use."""
    return {i: (landmarks[i].x * w, landmarks[i].y * h) for i in USED_LANDMARKS}
