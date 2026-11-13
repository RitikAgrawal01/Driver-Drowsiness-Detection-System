"""Alarm state machine (seconds, hysteresis) and PSI drift."""

import numpy as np
import pytest

from ddd_core.alarm import AlarmStateMachine
from ddd_core.drift import psi, psi_all


def test_alarm_turns_on_only_after_sustained_drowsiness_and_fires_once():
    sm = AlarmStateMachine(on_s=4, off_s=6)
    edges = [sm.update(t, True) for t in [0, 2, 4, 6, 8]]
    assert edges == [False, False, True, False, False]  # edge-triggered: one alarm event
    assert sm.alarm


def test_alarm_ignores_short_drowsy_blips():
    sm = AlarmStateMachine(on_s=4, off_s=6)
    for t, d in [(0, True), (2, True), (3, False), (4, True), (6, True)]:
        sm.update(t, d)
    assert not sm.alarm  # the alert at t=3 reset the streak


def test_alarm_hysteresis_needs_sustained_alert_to_clear():
    sm = AlarmStateMachine(on_s=0, off_s=6)
    sm.update(0, True)
    for t in [2, 4]:
        sm.update(t, False)
    assert sm.alarm  # only 2 s of alert so far
    sm.update(8, False)
    assert not sm.alarm


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    edges = np.quantile(base, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -1e308, 1e308
    props = np.histogram(base, bins=edges)[0] / len(base)
    assert psi(rng.normal(0, 1, 2000), edges, props) < 0.05
    assert psi(rng.normal(1.5, 1, 2000), edges, props) > 0.25


def test_psi_handles_nan_and_empty():
    edges, props = [-1e308, 0, 1e308], [0.5, 0.5]
    assert np.isnan(psi(np.array([np.nan]), edges, props))
    out = psi_all([{"f": -1.0}, {"f": 1.0}], {"f": {"edges": edges, "proportions": props}})
    assert out["f"] == pytest.approx(0.0, abs=1e-9)
