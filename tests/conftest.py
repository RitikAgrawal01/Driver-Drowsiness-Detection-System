"""
tests/conftest.py
Shared pytest fixtures for unit and integration tests.
Full fixtures added in Phase 5.
"""
import pytest


@pytest.fixture
def sample_features():
    """Returns a sample feature dict representing one sliding window."""
    return {
        "ear_mean": 0.28,
        "ear_min": 0.20,
        "perclos": 0.15,
        "mar_mean": 0.45,
        "head_pitch": 5.0,
        "head_yaw": -3.0,
        "head_roll": 1.5,
    }
