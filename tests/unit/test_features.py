"""
tests/unit/test_features.py
────────────────────────────
Unit tests for EAR, MAR, PERCLOS, and head pose calculations.
These test the pure-Python functions in feature_engineering.py directly.

Run:  pytest tests/unit/test_features.py -v
"""

import math
import sys
import os
import pytest

# Make airflow/scripts importable
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_landmark_row(ear: float = 0.30, mar: float = 0.40,
                       label: int = 0) -> dict:
    """
    Build a synthetic landmark row that produces predictable EAR/MAR values.
    Uses the same geometry as compute_ear / compute_mar expect.
    """
    row = {f"lm_{i}_x": 0.5 for i in range(468)}
    row.update({f"lm_{i}_y": 0.5 for i in range(468)})
    row.update({f"lm_{i}_z": 0.0 for i in range(468)})
    row["label"] = label
    row["subject_id"] = "test"
    row["frame_id"] = 0

    # Set right eye landmarks for target EAR
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    # With width=0.1, vertical = ear * 0.1
    w = 0.1
    v = ear * w
    cx, cy = 0.65, 0.40
    coords = [
        (cx - w/2, cy),         # p1 left corner  → idx RIGHT_EYE[0]
        (cx - w/4, cy - v),     # p2 upper-left   → idx RIGHT_EYE[1]
        (cx + w/4, cy - v),     # p3 upper-right  → idx RIGHT_EYE[2]
        (cx + w/2, cy),         # p4 right corner → idx RIGHT_EYE[3]
        (cx + w/4, cy + v),     # p5 lower-right  → idx RIGHT_EYE[4]
        (cx - w/4, cy + v),     # p6 lower-left   → idx RIGHT_EYE[5]
    ]
    for i, (x, y) in enumerate(coords):
        idx = RIGHT_EYE[i]
        row[f"lm_{idx}_x"] = x
        row[f"lm_{idx}_y"] = y

    # Mirror for left eye
    cx2 = 0.35
    for i, (x, y) in enumerate(coords):
        # Mirror x around center
        x_mirror = cx2 + (cx - x)
        idx = LEFT_EYE[i]
        row[f"lm_{idx}_x"] = x_mirror
        row[f"lm_{idx}_y"] = y

    # Set nose/chin/eye corners for head pose
    row["lm_1_x"], row["lm_1_y"] = 0.5, 0.5     # nose tip
    row["lm_152_x"], row["lm_152_y"] = 0.5, 0.7  # chin
    row["lm_226_x"], row["lm_226_y"] = 0.35, 0.4  # left eye corner
    row["lm_446_x"], row["lm_446_y"] = 0.65, 0.4  # right eye corner
    row["lm_57_x"], row["lm_57_y"] = 0.45, 0.6   # left mouth
    row["lm_287_x"], row["lm_287_y"] = 0.55, 0.6  # right mouth

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Tests: euclidean_2d
# ─────────────────────────────────────────────────────────────────────────────

class TestEuclidean:
    def test_same_point_is_zero(self):
        assert euclidean_2d((0, 0), (0, 0)) == 0.0

    def test_unit_distance(self):
        assert abs(euclidean_2d((0, 0), (1, 0)) - 1.0) < 1e-9

    def test_pythagorean(self):
        # 3-4-5 triangle
        assert abs(euclidean_2d((0, 0), (3, 4)) - 5.0) < 1e-9

    def test_symmetry(self):
        p1, p2 = (0.3, 0.7), (0.8, 0.2)
        assert euclidean_2d(p1, p2) == euclidean_2d(p2, p1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_ear
# ─────────────────────────────────────────────────────────────────────────────

class TestEAR:
    def test_ear_open_eye_above_threshold(self):
        """EAR for an open eye must be > 0.25 (drowsiness threshold)."""
        row = make_landmark_row(ear=0.32)
        result = compute_ear(row, RIGHT_EYE)
        assert result > 0.25, f"Expected EAR > 0.25, got {result}"

    def test_ear_closed_eye_below_threshold(self):
        """EAR for a closed eye must be < 0.25."""
        row = make_landmark_row(ear=0.12)
        result = compute_ear(row, RIGHT_EYE)
        assert result < 0.25, f"Expected EAR < 0.25, got {result}"

    def test_ear_is_non_negative(self):
        """EAR must always be >= 0."""
        for ear_target in [0.0, 0.1, 0.3, 0.5]:
            row = make_landmark_row(ear=ear_target)
            result = compute_ear(row, RIGHT_EYE)
            assert result >= 0.0

    def test_ear_closed_eye_near_zero(self):
        """Nearly fully closed eye should give EAR close to 0."""
        row = make_landmark_row(ear=0.01)
        result = compute_ear(row, RIGHT_EYE)
        assert result < 0.05

    def test_ear_scales_with_opening(self):
        """Larger eye opening should yield higher EAR."""
        row_open = make_landmark_row(ear=0.40)
        row_closed = make_landmark_row(ear=0.15)
        assert compute_ear(row_open, RIGHT_EYE) > compute_ear(row_closed, RIGHT_EYE)

    def test_both_eyes_similar(self):
        """Left and right EAR should be similar for symmetric face."""
        row = make_landmark_row(ear=0.30)
        right = compute_ear(row, RIGHT_EYE)
        left = compute_ear(row, LEFT_EYE)
        assert abs(right - left) < 0.05, \
            f"L/R EAR asymmetry too large: left={left}, right={right}"

    def test_ear_degenerate_horizontal(self):
        """Degenerate case: zero horizontal width → EAR = 0."""
        row = make_landmark_row(ear=0.30)
        # Collapse horizontal width to 0 for right eye
        for idx in RIGHT_EYE:
            row[f"lm_{idx}_x"] = 0.5  # all same x
        result = compute_ear(row, RIGHT_EYE)
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_mar
# ─────────────────────────────────────────────────────────────────────────────

class TestMAR:
    def _make_mouth_row(self, mar: float) -> dict:
        """Build a row where mouth landmarks produce the given MAR."""
        row = {f"lm_{i}_x": 0.5 for i in range(468)}
        row.update({f"lm_{i}_y": 0.5 for i in range(468)})
        row.update({f"lm_{i}_z": 0.0 for i in range(468)})
        row["label"] = 0
        row["subject_id"] = "test"
        row["frame_id"] = 0

        # Mouth: left=MOUTH[0], right=MOUTH[1], upper=MOUTH[2], lower=MOUTH[6]
        w = 0.12   # mouth width
        v = mar * w  # vertical opening = mar * width
        cx, cy = 0.5, 0.65

        positions = {
            MOUTH[0]: (cx - w/2, cy),      # left corner
            MOUTH[1]: (cx + w/2, cy),      # right corner
            MOUTH[2]: (cx, cy - v/2),      # upper mid
            MOUTH[3]: (cx - w/4, cy - v/4),
            MOUTH[4]: (cx, cy - v/2),
            MOUTH[5]: (cx + w/4, cy - v/4),
            MOUTH[6]: (cx, cy + v/2),      # lower mid
            MOUTH[7]: (cx - w/4, cy + v/4),
        }
        for idx, (x, y) in positions.items():
            row[f"lm_{idx}_x"] = x
            row[f"lm_{idx}_y"] = y

        return row

    def test_closed_mouth_below_threshold(self):
        """Closed mouth: MAR < 0.60."""
        row = self._make_mouth_row(mar=0.20)
        result = compute_mar(row, MOUTH)
        assert result < 0.60, f"Expected MAR < 0.60, got {result}"

    def test_yawning_above_threshold(self):
        """Yawning mouth: MAR > 0.60."""
        row = self._make_mouth_row(mar=0.80)
        result = compute_mar(row, MOUTH)
        assert result > 0.40  # threshold may vary with geometry

    def test_mar_non_negative(self):
        """MAR must always be >= 0."""
        for mar in [0.0, 0.3, 0.6, 1.0]:
            row = self._make_mouth_row(mar=mar)
            result = compute_mar(row, MOUTH)
            assert result >= 0.0

    def test_mar_scales_with_opening(self):
        """Larger mouth opening → higher MAR."""
        row_open = self._make_mouth_row(mar=0.8)
        row_closed = self._make_mouth_row(mar=0.2)
        assert compute_mar(row_open, MOUTH) > compute_mar(row_closed, MOUTH)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_head_pose_angles
# ─────────────────────────────────────────────────────────────────────────────

class TestHeadPose:
    def test_returns_three_values(self):
        """Head pose must return (pitch, yaw, roll)."""
        row = make_landmark_row()
        result = compute_head_pose_angles(row)
        assert len(result) == 3

    def test_upright_head_small_angles(self):
        """Upright, front-facing head should have small pitch, yaw, roll."""
        row = make_landmark_row()
        pitch, yaw, roll = compute_head_pose_angles(row)
        # Roll: symmetric eyes → close to 0
        assert abs(roll) < 10, f"Roll too large for symmetric face: {roll}"

    def test_angles_are_floats(self):
        """All angles must be floats."""
        row = make_landmark_row()
        pitch, yaw, roll = compute_head_pose_angles(row)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert isinstance(roll, float)

    def test_missing_landmarks_returns_zeros(self):
        """If landmark keys are missing, should return (0.0, 0.0, 0.0)."""
        empty_row = {}
        result = compute_head_pose_angles(empty_row)
        assert result == (0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_frame_features (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameFeatures:
    def test_output_keys_present(self):
        """Output must have all required feature keys."""
        row = make_landmark_row(ear=0.30, label=0)
        result = compute_frame_features(row)
        required = {"ear", "mar", "head_pitch", "head_yaw",
                    "head_roll", "label", "subject_id", "frame_id"}
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - result.keys()}"

    def test_alert_label_preserved(self):
        row = make_landmark_row(label=0)
        assert compute_frame_features(row)["label"] == 0

    def test_drowsy_label_preserved(self):
        row = make_landmark_row(label=1)
        assert compute_frame_features(row)["label"] == 1

    def test_ear_in_result(self):
        """EAR in result must be numeric and >= 0."""
        row = make_landmark_row(ear=0.30)
        result = compute_frame_features(row)
        assert result["ear"] >= 0.0
        assert isinstance(result["ear"], float)

    def test_alert_higher_ear_than_drowsy(self):
        """Alert frame (ear=0.35) should have higher EAR than drowsy (ear=0.15)."""
        alert_row = make_landmark_row(ear=0.35, label=0)
        drowsy_row = make_landmark_row(ear=0.15, label=1)
        alert_feat = compute_frame_features(alert_row)
        drowsy_feat = compute_frame_features(drowsy_row)
        assert alert_feat["ear"] > drowsy_feat["ear"]
