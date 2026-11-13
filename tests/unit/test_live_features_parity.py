"""
Training/serving parity (v1 bug M1): the backend's live extractor must produce exactly the
window features the training pipeline computes for the same frames.
"""

import numpy as np
import pytest

from backend.services.live_features import FrameDecodeError, LiveFeatureExtractor, decode_frame
from ddd_core.windows import ALL_FEATURES, window_features

CFG = {"ear_closed": 0.2, "long_closure_s": 0.5, "min_face_ratio": 0.8}


def synthetic_stream(seconds=40, fps=10, seed=0):
    rng = np.random.default_rng(seed)
    n = seconds * fps
    t = np.round(np.arange(n) / fps, 3)
    face = (rng.random(n) > 0.05).astype(int)
    ear = np.where(rng.random(n) < 0.1, 0.12, 0.3) + rng.normal(0, 0.01, n)
    mar, pitch, yaw, roll = rng.random(n) * 0.2, rng.normal(0, 5, n), rng.normal(0, 5, n), rng.normal(0, 3, n)
    for arr in (ear, mar, pitch, yaw, roll):
        arr[face == 0] = np.nan
    return t, face, ear, mar, pitch, yaw, roll


def test_live_window_equals_pipeline_window():
    t, face, ear, mar, pitch, yaw, roll = synthetic_stream()
    L, fps = 15.0, 10.0
    live = LiveFeatureExtractor(window_s=L, fps=fps, max_side=960, feature_cfg=CFG)
    for i in range(len(t)):
        live.add_frame(
            t[i], {"face": face[i], "ear": ear[i], "mar": mar[i], "pitch": pitch[i], "yaw": yaw[i], "roll": roll[i]}
        )
    now = t[-1]
    served = live.compute_window(now)

    # pipeline/window_features.py selects frames with np.searchsorted(t, [s, s + L]) -> [s, s + L)
    s = now - L + 1 / fps
    lo, hi = np.searchsorted(t, [s - 1e-9, s + L - 1e-9])
    trained = window_features(
        t[lo:hi],
        face[lo:hi],
        ear[lo:hi],
        mar[lo:hi],
        pitch[lo:hi],
        yaw[lo:hi],
        roll[lo:hi],
        length_s=L,
        cfg=CFG,
        fps=fps,
    )
    assert served is not None and trained is not None
    for k in ALL_FEATURES + ["face_ratio"]:
        assert served[k] == pytest.approx(trained[k], rel=1e-12, abs=1e-12), k


def test_sampler_keeps_target_fps_from_a_faster_camera():
    live = LiveFeatureExtractor(window_s=15, fps=10, max_side=960, feature_cfg=CFG)
    accepted = sum(live.should_sample(i / 30) for i in range(300))  # 10 s at 30 fps
    assert accepted == pytest.approx(100, abs=1)


def test_window_ready_warmup_and_face_tracking():
    live = LiveFeatureExtractor(window_s=15, fps=10, max_side=960, feature_cfg=CFG)
    assert not live.window_ready(0) and live.warmup_progress(0) == 0
    live.add_frame(0.0, {"face": 1, "ear": 0.3, "mar": 0, "pitch": 0, "yaw": 0, "roll": 0})
    live.add_frame(5.0, {"face": 0, "ear": np.nan, "mar": np.nan, "pitch": np.nan, "yaw": np.nan, "roll": np.nan})
    assert live.seconds_without_face(5.0) == pytest.approx(5.0)
    assert live.warmup_progress(7.5) == pytest.approx(0.5)
    assert live.window_ready(15.0)


def test_decode_frame(jpeg_b64):
    assert decode_frame(jpeg_b64).shape == (48, 64, 3)
    assert decode_frame("data:image/jpeg;base64," + jpeg_b64).shape == (48, 64, 3)
    with pytest.raises(FrameDecodeError):
        decode_frame("not-base64!!")
    with pytest.raises(FrameDecodeError):
        decode_frame("aGVsbG8=")  # valid base64, not an image
