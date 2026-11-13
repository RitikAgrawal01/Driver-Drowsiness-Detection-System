"""Time-based window features: the 9 v1 features (bug-fixed) + 3 blink features.

A window is a span of SECONDS (not a count of frames), so it means the same thing at any
frame rate. The training pipeline slides it over recorded videos; the backend computes the
most recent window of a live stream. Both call `window_features`, so they cannot differ.
"""

import numpy as np

BASE_FEATURES = [
    "ear_mean",
    "ear_min",
    "ear_std",
    "perclos",
    "mar_mean",
    "mar_max",
    "head_pitch_mean",
    "head_yaw_mean",
    "head_roll_mean",
]
BLINK_FEATURES = ["blink_rate_pm", "blink_dur_mean_s", "long_closure_rate_pm"]
POSE_FEATURES = ["head_pitch_mean", "head_yaw_mean", "head_roll_mean"]
ALL_FEATURES = BASE_FEATURES + BLINK_FEATURES

FEATURE_SETS = {
    "v1_9_fixed": BASE_FEATURES,
    "v2_12": ALL_FEATURES,
    "v2_12_no_pose": [f for f in ALL_FEATURES if f not in POSE_FEATURES],
}


def circular_mean_deg(a: np.ndarray) -> float:
    """Mean of angles that wraps correctly: mean(-179, 179) = 180, not 0."""
    a = np.asarray(a, dtype=float)
    a = np.radians(a[~np.isnan(a)])
    if a.size == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def closure_events(t: np.ndarray, closed: np.ndarray, step: float) -> np.ndarray:
    """Runs of consecutive closed frames -> array of (start_t, duration_s).
    A run breaks at an open frame, a no-face frame (closed=False) or a time gap > 1.5 steps."""
    events, start, last = [], None, None
    for ti, c in zip(t, closed):
        if c and start is not None and ti - last <= 1.5 * step:
            last = ti
            continue
        if start is not None:
            events.append((start, last - start + step))
            start = None
        if c:
            start = last = ti
    if start is not None:
        events.append((start, last - start + step))
    return np.array(events, dtype=float).reshape(-1, 2)


def window_features(t, face, ear, mar, pitch, yaw, roll, length_s: float, cfg: dict, fps: float) -> dict | None:
    """Features for ONE window given its frames (already restricted to the window, time-sorted).

    cfg = params['features'] (ear_closed, long_closure_s, min_face_ratio).
    Returns None when the face was visible in fewer than min_face_ratio of the expected frames.
    """
    t = np.asarray(t, dtype=float)
    face = np.asarray(face).astype(bool)
    expected = length_s * fps
    if face.sum() < cfg["min_face_ratio"] * expected:
        return None
    step = 1.0 / fps
    ear, mar = np.asarray(ear, dtype=float), np.asarray(mar, dtype=float)
    e = ear[face]
    closed = face & (np.nan_to_num(ear, nan=1.0) < cfg["ear_closed"])
    ev = closure_events(t, closed, step)
    n_long = int((ev[:, 1] >= cfg["long_closure_s"]).sum()) if len(ev) else 0
    minutes = length_s / 60.0
    return {
        "ear_mean": float(e.mean()),
        "ear_min": float(e.min()),
        "ear_std": float(e.std()),
        "perclos": float((e < cfg["ear_closed"]).mean()),
        "mar_mean": float(np.nanmean(mar[face])),
        "mar_max": float(np.nanmax(mar[face])),
        "head_pitch_mean": circular_mean_deg(np.asarray(pitch, dtype=float)[face]),
        "head_yaw_mean": circular_mean_deg(np.asarray(yaw, dtype=float)[face]),
        "head_roll_mean": circular_mean_deg(np.asarray(roll, dtype=float)[face]),
        "blink_rate_pm": (len(ev) - n_long) / minutes,
        "blink_dur_mean_s": float(ev[:, 1].mean()) if len(ev) else 0.0,
        "long_closure_rate_pm": n_long / minutes,
        "face_ratio": float(face.sum() / expected),
    }
