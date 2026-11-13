"""ddd_core.geometry — the per-frame maths shared by training and serving."""

import math

import numpy as np
import pytest

from ddd_core.geometry import (
    FACE_3D,
    LEFT_EYE,
    POSE,
    RIGHT_EYE,
    eye_aspect_ratio,
    frame_features,
    head_pose,
    mouth_aspect_ratio,
)
from tests.conftest import eye_points, face_points


def as_dict(points, idx):
    return dict(zip(idx, points))


@pytest.mark.parametrize("ear", [0.05, 0.2, 0.3, 0.45])
def test_ear_recovers_constructed_value(ear):
    assert eye_aspect_ratio(as_dict(eye_points(ear), RIGHT_EYE), RIGHT_EYE) == pytest.approx(ear)


def test_ear_is_scale_invariant_when_x_and_y_scale_equally():
    pts = eye_points(0.28)
    scaled = [(3 * x, 3 * y) for x, y in pts]
    assert eye_aspect_ratio(as_dict(scaled, LEFT_EYE), LEFT_EYE) == pytest.approx(0.28)


def test_ear_changes_with_unequal_scaling_which_is_why_real_width_and_height_matter():
    """v1 forced 640x480 on every video; scaling x and y differently changes EAR (bug C4)."""
    pts = eye_points(0.28)
    squashed = [(x * 1.0, y * 0.5) for x, y in pts]
    assert eye_aspect_ratio(as_dict(squashed, RIGHT_EYE), RIGHT_EYE) == pytest.approx(0.14)


def test_ear_degenerate_eye_returns_nan():
    pts = {i: (10.0, 10.0) for i in RIGHT_EYE}
    assert math.isnan(eye_aspect_ratio(pts, RIGHT_EYE))


def test_mar_is_zero_for_closed_mouth_with_inner_lips():
    assert mouth_aspect_ratio(face_points(mouth_open=0.0)) == pytest.approx(0.0)
    assert mouth_aspect_ratio(face_points(mouth_open=0.5)) == pytest.approx(0.5)


def test_head_pose_of_upright_frontal_face_is_near_zero():
    """v1's y-up model against y-down image points gave pitch ~ +/-180 for an upright face."""
    pitch, yaw, roll = head_pose(face_points(), 640, 480)
    assert abs(pitch) < 2 and abs(yaw) < 2 and abs(roll) < 2


def test_head_pose_recovers_a_known_yaw():
    import cv2

    yaw = math.radians(20)
    R = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    cam = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=float)
    img, _ = cv2.projectPoints(FACE_3D, cv2.Rodrigues(R)[0], np.array([0.0, 0.0, 600.0]), cam, np.zeros(4))
    pts = dict(zip(POSE, img.reshape(-1, 2)))
    _, est_yaw, _ = head_pose(pts, 640, 480)
    assert est_yaw == pytest.approx(20, abs=1.5)


def test_frame_features_without_face_is_flagged_not_zeroed():
    f = frame_features(None, 640, 480)
    assert f["face"] == 0
    assert all(math.isnan(f[k]) for k in ["ear", "mar", "pitch", "yaw", "roll"])


def test_frame_features_with_face():
    f = frame_features(face_points(ear=0.3), 640, 480)
    assert f["face"] == 1
    assert f["ear"] == pytest.approx(0.3)
