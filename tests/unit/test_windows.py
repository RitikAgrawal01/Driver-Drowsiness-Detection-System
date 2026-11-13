"""ddd_core.windows — window features (the 9 fixed + 3 blink features)."""

import numpy as np
import pytest

from ddd_core.windows import (
    ALL_FEATURES,
    FEATURE_SETS,
    POSE_FEATURES,
    circular_mean_deg,
    closure_events,
    window_features,
)

CFG = {"ear_closed": 0.2, "long_closure_s": 0.5, "min_face_ratio": 0.8}
FPS = 10.0


def frames(ear, face=None, pitch=0.0):
    n = len(ear)
    t = np.arange(n) / FPS
    face = np.ones(n, dtype=int) if face is None else np.asarray(face)
    z = np.zeros(n)
    return t, face, np.asarray(ear, float), z, np.full(n, pitch), z, z


def test_feature_sets():
    assert len(FEATURE_SETS["v2_12"]) == 12 and FEATURE_SETS["v2_12"] == ALL_FEATURES
    assert not set(POSE_FEATURES) & set(FEATURE_SETS["v2_12_no_pose"])


def test_circular_mean_wraps_correctly():
    """mean(-179, 179) is 180 degrees, not 0 (v1 used an arithmetic mean)."""
    assert abs(circular_mean_deg(np.array([-179.0, 179.0]))) == pytest.approx(180, abs=1e-6)
    assert circular_mean_deg(np.array([10.0, 20.0, np.nan])) == pytest.approx(15)


def test_closure_events_runs_and_gaps():
    t = np.arange(10) / FPS
    closed = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1], bool)
    ev = closure_events(t, closed, 0.1)
    assert ev[:, 1] == pytest.approx([0.2, 0.6])
    t_gap = np.array([0.0, 0.1, 0.5, 0.6])  # a time gap breaks a run
    assert len(closure_events(t_gap, np.ones(4, bool), 0.1)) == 2


def test_window_features_perclos_blinks_and_long_closures():
    ear = np.full(150, 0.3)
    ear[10:12] = 0.1  # 0.2 s blink
    ear[50:53] = 0.1  # 0.3 s blink
    ear[100:108] = 0.1  # 0.8 s long closure
    w = window_features(*frames(ear), length_s=15.0, cfg=CFG, fps=FPS)
    assert w["perclos"] == pytest.approx(13 / 150)
    assert w["blink_rate_pm"] == pytest.approx(2 / 0.25)
    assert w["long_closure_rate_pm"] == pytest.approx(1 / 0.25)
    assert w["blink_dur_mean_s"] == pytest.approx((0.2 + 0.3 + 0.8) / 3)
    assert w["ear_min"] == pytest.approx(0.1)
    assert w["ear_std"] == pytest.approx(np.std(ear))  # population std, same in training and serving


def test_window_dropped_when_face_mostly_missing():
    face = np.ones(150, dtype=int)
    face[:40] = 0  # 73 % visible < 80 %
    assert window_features(*frames(np.full(150, 0.3), face=face), length_s=15.0, cfg=CFG, fps=FPS) is None


def test_no_face_frames_are_excluded_not_counted_as_open_or_closed():
    ear = np.full(150, 0.3)
    face = np.ones(150, dtype=int)
    face[:20] = 0
    ear[:20] = np.nan
    w = window_features(*frames(ear, face=face), length_s=15.0, cfg=CFG, fps=FPS)
    assert w["perclos"] == 0 and w["face_ratio"] == pytest.approx(130 / 150)
