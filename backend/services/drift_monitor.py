"""
Global drift monitor: PSI of live window features vs the champion's training baseline.

v1 issues fixed: one buffer for ALL sessions (not a per-session gauge where the last writer
wins); only real model-input windows are added (one per prediction, not 6 cached duplicates
per frame); no score until `min_windows` windows are collected.
"""

import threading
from collections import deque

from backend.metrics import DRIFT_WINDOWS, FEATURE_PSI, OVERALL_DRIFT
from ddd_core.config import load_params
from ddd_core.drift import psi_all


class DriftMonitor:
    def __init__(self):
        cfg = load_params()["drift"]
        self.min_windows = int(cfg["min_windows"])
        self.every = 10
        self._buf: deque = deque(maxlen=int(cfg["buffer_windows"]))
        self._baseline: dict | None = None
        self._lock = threading.Lock()
        self._since_update = 0
        self.scores: dict = {}
        OVERALL_DRIFT.set(-1)

    def set_baseline(self, baseline: dict | None) -> None:
        with self._lock:
            self._baseline = baseline
            self._buf.clear()
            self.scores = {}
        OVERALL_DRIFT.set(-1)

    def add(self, window: dict) -> None:
        with self._lock:
            self._buf.append(window)
            self._since_update += 1
            n = len(self._buf)
            due = self._baseline is not None and n >= self.min_windows and self._since_update >= self.every
        DRIFT_WINDOWS.set(n)
        if due:
            self.compute()

    def compute(self) -> dict:
        with self._lock:
            if self._baseline is None or len(self._buf) < self.min_windows:
                return {}
            rows, baseline = list(self._buf), self._baseline
            self._since_update = 0
        scores = psi_all(rows, baseline)
        for feat, s in scores.items():
            FEATURE_PSI.labels(feature=feat).set(s)
        overall = max(scores.values()) if scores else -1
        OVERALL_DRIFT.set(overall)
        self.scores = {"overall": overall, "features": scores, "windows": len(rows)}
        return self.scores


drift_monitor = DriftMonitor()
