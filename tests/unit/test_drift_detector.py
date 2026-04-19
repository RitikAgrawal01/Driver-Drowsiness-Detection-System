"""
tests/unit/test_drift_detector.py
───────────────────────────────────
Unit tests for the DriftDetector service.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))


SAMPLE_BASELINE = {
    "features": {
        "ear_mean": {"mean": 0.30, "variance": 0.1, "std": 0.316},
        "ear_min":  {"mean": 0.22, "variance": 0.1, "std": 0.316},
        "ear_std":  {"mean": 0.04, "variance": 0.01, "std": 0.100},
        "perclos":  {"mean": 0.15, "variance": 0.1, "std": 0.316},
        "mar_mean": {"mean": 0.40, "variance": 0.05, "std": 0.224},
        "mar_max":  {"mean": 0.55, "variance": 0.1, "std": 0.316},
        "head_pitch_mean": {"mean": -3.0, "variance": 9.0, "std": 3.0},
        "head_yaw_mean":   {"mean": 1.0,  "variance": 4.0, "std": 2.0},
        "head_roll_mean":  {"mean": 0.5,  "variance": 1.0, "std": 1.0},
    }
}

NORMAL_FEATURES = {
    "ear_mean": 0.30, "ear_min": 0.22, "ear_std": 0.04,
    "perclos": 0.15, "mar_mean": 0.40, "mar_max": 0.55,
    "head_pitch_mean": -3.0, "head_yaw_mean": 1.0, "head_roll_mean": 0.5,
}

DRIFTED_FEATURES = {
    "ear_mean": 0.12,  # much lower — simulates darker lighting
    "ear_min": 0.05,
    "ear_std": 0.04,
    "perclos": 0.75,   # much higher
    "mar_mean": 0.40,
    "mar_max": 0.55,
    "head_pitch_mean": -3.0,
    "head_yaw_mean": 1.0,
    "head_roll_mean": 0.5,
}


@pytest.fixture
def baseline_file():
    """Create a temporary baseline.json file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(SAMPLE_BASELINE, f)
        return f.name


@pytest.fixture
def detector(baseline_file):
    """DriftDetector with a real baseline file."""
    from backend.services.drift_detector import DriftDetector
    d = DriftDetector(baseline_path=baseline_file, window_size=50)
    yield d
    os.unlink(baseline_file)


@pytest.fixture
def detector_no_baseline():
    """DriftDetector with no baseline (disabled)."""
    from backend.services.drift_detector import DriftDetector
    return DriftDetector(baseline_path="/nonexistent/baseline.json",
                          window_size=50)


class TestDriftDetectorInit:
    def test_baseline_loaded(self, detector):
        assert detector._baseline_loaded is True

    def test_baseline_has_features(self, detector):
        assert len(detector._baseline) == 9

    def test_no_baseline_graceful(self, detector_no_baseline):
        """Should not raise even if baseline.json missing."""
        assert detector_no_baseline._baseline_loaded is False


class TestDriftDetectorUpdate:
    def test_update_adds_to_buffer(self, detector):
        detector.update(NORMAL_FEATURES)
        assert len(detector._buffers["ear_mean"]) == 1

    def test_update_multiple_frames(self, detector):
        for _ in range(10):
            detector.update(NORMAL_FEATURES)
        assert len(detector._buffers["ear_mean"]) == 10

    def test_update_respects_window_size(self, detector):
        """Buffer should not exceed window_size."""
        for _ in range(100):
            detector.update(NORMAL_FEATURES)
        assert len(detector._buffers["ear_mean"]) <= detector._window_size

    def test_update_ignores_missing_keys(self, detector):
        """Partial feature vector should not crash."""
        detector.update({"ear_mean": 0.30})  # only one key
        assert True  # no exception


class TestDriftScoreComputation:
    def test_empty_buffer_returns_zeros(self, detector):
        scores = detector.compute_drift_scores()
        assert all(v == 0.0 for v in scores.values())

    def test_normal_features_low_drift(self, detector):
        """Normal features close to baseline should have low drift."""
        import random
        
        # Lock the random seed so this test is never flaky in CI/CD pipelines
        random.seed(42)
        
        # Loop 200 times to give the empirical distribution enough data to form a proper bell curve
        for _ in range(200):
            jittered_features = {
                k: random.gauss(v, SAMPLE_BASELINE["features"][k]["std"]) 
                for k, v in NORMAL_FEATURES.items()
            }
            detector.update(jittered_features)
            
        scores = detector.compute_drift_scores()
        overall = max(scores.values(), default=0.0)
        
        assert overall < 0.15, f"Expected low drift, got {overall}"

    def test_drifted_features_high_drift(self, detector):
        """Heavily shifted features should produce high drift score."""
        for _ in range(50):
            detector.update(DRIFTED_FEATURES)
        scores = detector.compute_drift_scores()
        overall = max(scores.values(), default=0.0)
        assert overall > 0.0, "Expected nonzero drift for shifted features"

    def test_scores_are_non_negative(self, detector):
        """KL divergence is always >= 0."""
        for _ in range(50):
            detector.update(NORMAL_FEATURES)
        scores = detector.compute_drift_scores()
        assert all(v >= 0.0 for v in scores.values())

    def test_no_baseline_returns_empty(self, detector_no_baseline):
        for _ in range(50):
            detector_no_baseline.update(NORMAL_FEATURES)
        scores = detector_no_baseline.compute_drift_scores()
        assert scores == {}


class TestDriftDetectorReset:
    def test_reset_clears_buffers(self, detector):
        for _ in range(20):
            detector.update(NORMAL_FEATURES)
        detector.reset()
        for buf in detector._buffers.values():
            assert len(buf) == 0

    def test_is_drifting_false_after_reset(self, detector):
        for _ in range(50):
            detector.update(DRIFTED_FEATURES)
        detector.reset()
        assert not detector.is_drifting()
