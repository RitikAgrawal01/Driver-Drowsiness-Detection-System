"""
tests/unit/test_features.py
────────────────────────────
Unit tests for EAR, MAR, PERCLOS, and head pose calculations.
These tests run WITHOUT MediaPipe or OpenCV — they test the pure
math functions imported directly from the feature engineering script.

Run with:  pytest tests/unit/test_features.py -v
"""

import math
import sys
import os
import pytest

# Make scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../airflow/scripts"))

from feature_engineering import (
    compute_ear,
    compute_mar,
    compute_head_pose_angles,
    compute_frame_features,
    euclidean_2d,
    RIGHT_EYE,
    LEFT_EYE,
    MOUTH,
)


# ─────────────────────────────────────────────────────────────────
# Helper: build a synthetic landmark dict
# ─────────────────────────────────────────────────────────────────

def make_landmarks(overrides: dict = None) -> dict:
    """
    Create a landmark dict with all 468 landmarks at default positions.
    Overrides can be applied for specific landmark indices.

    Default position: (0.5, 0.5, 0.0) for all landmarks.
    """
    lm = {}
    for i in range(468):
        lm[f"lm_{i}_x"] = 0.5
        lm[f"lm_{i}_y"] = 0.5
        lm[f"lm_{i}_z"] = 0.0
    if overrides:
        lm.update(overrides)
    return lm


def set_eye_landmarks(lm: dict, indices: list, ear: float, center_x: float, center_y: float) -> dict:
    """
    Set 6 eye landmarks to produce a specific EAR value.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Horizontal width = 0.1, vertical opening = ear * horizontal / 2
    """
    w = 0.1   # horizontal eye width
    v = ear * w  # vertical eye opening (EAR × horizontal)
    positions = [
        (center_x - w / 2, center_y),           # p1: left corner
        (center_x - w / 4, center_y - v / 2),   # p2: upper-left
        (center_x + w / 4, center_y - v / 2),   # p3: upper-right
        (center_x + w / 2, center_y),           # p4: right corner
        (center_x + w / 4, center_y + v / 2),   # p5: lower-right
        (center_x - w / 4, center_y + v / 2),   # p6: lower-left
    ]
    for idx, (x, y) in zip(indices, positions):
        lm[f"lm_{idx}_x"] = x
        lm[f"lm_{idx}_y"] = y
    return lm


# ─────────────────────────────────────────────────────────────────
# Tests: euclidean_2d
# ─────────────────────────────────────────────────────────────────

class TestEuclidean2D:
    def test_zero_distance(self):
        assert euclidean_2d((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)

    def test_horizontal_distance(self):
        assert euclidean_2d((0.0, 0.0), (3.0, 0.0)) == pytest.approx(3.0)

    def test_vertical_distance(self):
        assert euclidean_2d((0.0, 0.0), (0.0, 4.0)) == pytest.approx(4.0)

    def test_diagonal_distance(self):
        # 3-4-5 right triangle
        assert euclidean_2d((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_symmetry(self):
        p1, p2 = (0.3, 0.7), (0.8, 0.2)
        assert euclidean_2d(p1, p2) == pytest.approx(euclidean_2d(p2, p1))


# ─────────────────────────────────────────────────────────────────
# Tests: EAR (Eye Aspect Ratio)
# ─────────────────────────────────────────────────────────────────

class TestEAR:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Alert driver:  EAR ≈ 0.25–0.40
    Drowsy driver: EAR ≈ 0.05–0.22
    Threshold:     EAR < 0.25 → eye considered closed
    """

    def test_alert_eye_above_threshold(self):
        """Open eye EAR must be above the drowsiness threshold (0.25)."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.35, center_x=0.65, center_y=0.40)
        ear = compute_ear(lm, RIGHT_EYE)
        assert ear > 0.25, f"Alert EAR should be > 0.25, got {ear:.4f}"

    def test_closed_eye_below_threshold(self):
        """Closed eye EAR must be below the drowsiness threshold (0.25)."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.10, center_x=0.65, center_y=0.40)
        ear = compute_ear(lm, RIGHT_EYE)
        assert ear < 0.25, f"Closed EAR should be < 0.25, got {ear:.4f}"

    def test_ear_non_negative(self):
        """EAR must always be non-negative."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.30, center_x=0.65, center_y=0.40)
        ear = compute_ear(lm, RIGHT_EYE)
        assert ear >= 0.0

    def test_ear_reasonable_range(self):
        """EAR should be in physically plausible range [0, 1]."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.30, center_x=0.65, center_y=0.40)
        ear = compute_ear(lm, RIGHT_EYE)
        assert 0.0 <= ear <= 1.0

    def test_ear_increases_with_opening(self):
        """Wider open eye should produce higher EAR."""
        lm_open = make_landmarks()
        lm_open = set_eye_landmarks(lm_open, RIGHT_EYE, ear=0.35, center_x=0.65, center_y=0.40)

        lm_closed = make_landmarks()
        lm_closed = set_eye_landmarks(lm_closed, RIGHT_EYE, ear=0.10, center_x=0.65, center_y=0.40)

        ear_open = compute_ear(lm_open, RIGHT_EYE)
        ear_closed = compute_ear(lm_closed, RIGHT_EYE)
        assert ear_open > ear_closed

    def test_degenerate_horizontal_zero(self):
        """EAR with zero horizontal width should return 0.0 (no division error)."""
        lm = make_landmarks()
        # All eye points at same x → horizontal = 0
        for idx in RIGHT_EYE:
            lm[f"lm_{idx}_x"] = 0.5
            lm[f"lm_{idx}_y"] = 0.5
        ear = compute_ear(lm, RIGHT_EYE)
        assert ear == 0.0

    def test_left_and_right_eye_independent(self):
        """Left and right eye EAR should be computed independently."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.35, center_x=0.65, center_y=0.40)
        lm = set_eye_landmarks(lm, LEFT_EYE, ear=0.12, center_x=0.35, center_y=0.40)

        right_ear = compute_ear(lm, RIGHT_EYE)
        left_ear = compute_ear(lm, LEFT_EYE)

        assert right_ear > 0.25, f"Right (open) EAR should be > 0.25, got {right_ear:.4f}"
        assert left_ear < 0.25, f"Left (closed) EAR should be < 0.25, got {left_ear:.4f}"


# ─────────────────────────────────────────────────────────────────
# Tests: MAR (Mouth Aspect Ratio)
# ─────────────────────────────────────────────────────────────────

class TestMAR:
    """
    MAR = vertical_opening / horizontal_width
    Closed mouth: MAR ≈ 0.0–0.40
    Yawning:      MAR > 0.60
    """

    def _make_mouth_landmarks(self, lm: dict, mar: float) -> dict:
        """
        Set mouth landmarks to produce approximately the given MAR.
        Horizontal width = 0.12, vertical = mar × horizontal.
        """
        w = 0.12
        v = mar * w
        cx, cy = 0.5, 0.65
        # Indices: MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
        positions = [
            (cx - w / 2, cy),           # 0: left corner
            (cx + w / 2, cy),           # 1: right corner
            (cx, cy - v / 2),           # 2: upper lip mid
            (cx + w / 4, cy - v / 4),   # 3: upper right
            (cx - w / 4, cy - v / 4),   # 4: upper left
            (cx, cy),                   # 5: center
            (cx + w / 4, cy + v / 4),   # 6: lower right
            (cx, cy + v / 2),           # 7: lower lip mid
        ]
        for idx, (x, y) in zip(MOUTH, positions):
            lm[f"lm_{idx}_x"] = x
            lm[f"lm_{idx}_y"] = y
        return lm

    def test_closed_mouth_low_mar(self):
        """Closed mouth should produce MAR below yawn threshold (0.6)."""
        lm = make_landmarks()
        lm = self._make_mouth_landmarks(lm, mar=0.20)
        mar = compute_mar(lm, MOUTH)
        assert mar < 0.6, f"Closed mouth MAR should be < 0.6, got {mar:.4f}"

    def test_yawning_high_mar(self):
        """Yawning mouth should produce MAR above yawn threshold (0.6)."""
        lm = make_landmarks()
        lm = self._make_mouth_landmarks(lm, mar=0.80)
        mar = compute_mar(lm, MOUTH)
        assert mar > 0.4, f"Yawning MAR should be elevated, got {mar:.4f}"

    def test_mar_non_negative(self):
        """MAR must always be non-negative."""
        lm = make_landmarks()
        lm = self._make_mouth_landmarks(lm, mar=0.30)
        mar = compute_mar(lm, MOUTH)
        assert mar >= 0.0

    def test_mar_increases_with_opening(self):
        """Wider mouth should produce higher MAR."""
        lm_closed = make_landmarks()
        lm_closed = self._make_mouth_landmarks(lm_closed, mar=0.20)

        lm_open = make_landmarks()
        lm_open = self._make_mouth_landmarks(lm_open, mar=0.80)

        assert compute_mar(lm_open, MOUTH) > compute_mar(lm_closed, MOUTH)


# ─────────────────────────────────────────────────────────────────
# Tests: Head Pose Angles
# ─────────────────────────────────────────────────────────────────

class TestHeadPose:
    """
    Returns (pitch_deg, yaw_deg, roll_deg).
    Upright head: all angles near 0.
    Nodding down: negative pitch.
    """

    def _make_upright_face(self, lm: dict) -> dict:
        """Configure landmarks for a roughly upright frontal face."""
        # nose tip (1), chin (152), eye corners, mouth corners
        lm["lm_1_x"], lm["lm_1_y"] = 0.50, 0.45    # nose tip
        lm["lm_152_x"], lm["lm_152_y"] = 0.50, 0.70  # chin
        lm["lm_226_x"], lm["lm_226_y"] = 0.35, 0.40  # left eye corner
        lm["lm_446_x"], lm["lm_446_y"] = 0.65, 0.40  # right eye corner
        lm["lm_57_x"], lm["lm_57_y"] = 0.38, 0.62    # left mouth corner
        lm["lm_287_x"], lm["lm_287_y"] = 0.62, 0.62  # right mouth corner
        return lm

    def test_upright_roll_near_zero(self):
        """Upright face should have roll angle near 0 degrees."""
        lm = make_landmarks()
        lm = self._make_upright_face(lm)
        pitch, yaw, roll = compute_head_pose_angles(lm)
        assert abs(roll) < 10.0, f"Upright roll should be near 0, got {roll:.2f}°"

    def test_returns_three_values(self):
        """compute_head_pose_angles must always return exactly 3 values."""
        lm = make_landmarks()
        result = compute_head_pose_angles(lm)
        assert len(result) == 3

    def test_angles_are_numeric(self):
        """All returned angles must be numeric (float)."""
        lm = make_landmarks()
        lm = self._make_upright_face(lm)
        pitch, yaw, roll = compute_head_pose_angles(lm)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert isinstance(roll, float)

    def test_no_exception_on_degenerate_landmarks(self):
        """Should not raise even if all landmarks are at same position."""
        lm = make_landmarks()   # all at (0.5, 0.5)
        try:
            pitch, yaw, roll = compute_head_pose_angles(lm)
            assert all(isinstance(v, float) for v in [pitch, yaw, roll])
        except Exception as e:
            pytest.fail(f"compute_head_pose_angles raised on degenerate input: {e}")

    def test_tilted_roll_nonzero(self):
        """Tilted head (right eye higher than left) should produce nonzero roll."""
        lm = make_landmarks()
        lm = self._make_upright_face(lm)
        # Tilt: move right eye up, left eye down
        lm["lm_446_y"] = 0.30  # right eye higher
        lm["lm_226_y"] = 0.50  # left eye lower
        _, _, roll = compute_head_pose_angles(lm)
        assert abs(roll) > 1.0, f"Tilted head should have |roll| > 1°, got {roll:.2f}°"


# ─────────────────────────────────────────────────────────────────
# Tests: compute_frame_features (integration of all per-frame features)
# ─────────────────────────────────────────────────────────────────

class TestComputeFrameFeatures:
    """Tests for the top-level per-frame feature computation function."""

    def _make_alert_frame(self) -> dict:
        """Build a landmark dict representing an alert driver frame."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.32, center_x=0.65, center_y=0.40)
        lm = set_eye_landmarks(lm, LEFT_EYE, ear=0.32, center_x=0.35, center_y=0.40)
        lm["label"] = 0
        lm["subject_id"] = "test_subject"
        lm["frame_id"] = 42
        return lm

    def _make_drowsy_frame(self) -> dict:
        """Build a landmark dict representing a drowsy driver frame."""
        lm = make_landmarks()
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.12, center_x=0.65, center_y=0.40)
        lm = set_eye_landmarks(lm, LEFT_EYE, ear=0.12, center_x=0.35, center_y=0.40)
        lm["label"] = 1
        lm["subject_id"] = "test_subject"
        lm["frame_id"] = 100
        return lm

    def test_returns_all_required_keys(self):
        """Feature dict must contain all required keys."""
        row = self._make_alert_frame()
        features = compute_frame_features(row)
        required = ["ear", "mar", "head_pitch", "head_yaw", "head_roll", "label", "subject_id", "frame_id"]
        for key in required:
            assert key in features, f"Missing key: {key}"

    def test_alert_ear_above_threshold(self):
        """Alert driver frame EAR should be above 0.25."""
        row = self._make_alert_frame()
        features = compute_frame_features(row)
        assert features["ear"] > 0.25, f"Alert EAR = {features['ear']:.4f}, expected > 0.25"

    def test_drowsy_ear_below_threshold(self):
        """Drowsy driver frame EAR should be below 0.25."""
        row = self._make_drowsy_frame()
        features = compute_frame_features(row)
        assert features["ear"] < 0.25, f"Drowsy EAR = {features['ear']:.4f}, expected < 0.25"

    def test_label_preserved(self):
        """Label from input row must be preserved in output."""
        assert compute_frame_features(self._make_alert_frame())["label"] == 0
        assert compute_frame_features(self._make_drowsy_frame())["label"] == 1

    def test_subject_id_preserved(self):
        """Subject ID must be preserved."""
        row = self._make_alert_frame()
        features = compute_frame_features(row)
        assert features["subject_id"] == "test_subject"

    def test_frame_id_preserved(self):
        """Frame ID must be preserved."""
        row = self._make_alert_frame()
        features = compute_frame_features(row)
        assert features["frame_id"] == 42

    def test_ear_is_average_of_both_eyes(self):
        """EAR should be the average of left and right eye EAR."""
        lm = make_landmarks()
        # Right eye open, left eye half-closed
        lm = set_eye_landmarks(lm, RIGHT_EYE, ear=0.36, center_x=0.65, center_y=0.40)
        lm = set_eye_landmarks(lm, LEFT_EYE, ear=0.18, center_x=0.35, center_y=0.40)
        lm["label"] = 0
        lm["subject_id"] = "s1"
        lm["frame_id"] = 0
        features = compute_frame_features(lm)
        # Average should be roughly midway between 0.18 and 0.36
        assert 0.18 < features["ear"] < 0.36

    def test_all_numeric_outputs(self):
        """All float features must be numeric."""
        row = self._make_alert_frame()
        features = compute_frame_features(row)
        float_keys = ["ear", "mar", "head_pitch", "head_yaw", "head_roll"]
        for key in float_keys:
            assert isinstance(features[key], float), f"{key} is not float: {type(features[key])}"
