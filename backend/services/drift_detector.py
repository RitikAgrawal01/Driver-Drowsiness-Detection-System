"""
backend/services/drift_detector.py
────────────────────────────────────
Real-time data drift detector.

Loads training baseline statistics from data/features/baseline.json.
Maintains a rolling buffer of live feature values per session.
Computes KL-divergence between live distribution and training baseline.
Emits per-feature and overall drift scores to Prometheus gauges.

Drift is detected when KL-divergence exceeds DRIFT_SCORE_THRESHOLD (0.15).
"""

import json
import math
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import FEATURE_DRIFT_SCORE, OVERALL_DRIFT_SCORE

logger = get_logger("drift_detector")

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]


class DriftDetector:
    """
    Session-scoped drift detector.
    One instance per active monitoring session.
    """

    def __init__(self, baseline_path: Optional[str] = None,
                 window_size: int = 300):
        """
        Args:
            baseline_path: path to baseline.json from feature engineering
            window_size:   number of feature vectors to keep in rolling buffer
                           (default 300 = ~10 seconds at 30fps)
        """
        settings = get_settings()
        self._baseline_path = baseline_path or settings.baseline_json_path
        self._window_size = window_size
        self._threshold = settings.drift_score_threshold

        # Per-feature rolling buffers
        self._buffers: Dict[str, deque] = {
            col: deque(maxlen=window_size) for col in FEATURE_COLS
        }

        # Baseline stats loaded once
        self._baseline: Dict[str, dict] = {}
        self._baseline_loaded = False

        self._load_baseline()

    def _load_baseline(self):
        """Load baseline.json from the feature engineering stage."""
        if not os.path.isfile(self._baseline_path):
            logger.warning(
                f"Baseline file not found: {self._baseline_path}. "
                f"Drift detection disabled until baseline is available."
            )
            return

        try:
            with open(self._baseline_path) as f:
                data = json.load(f)
            self._baseline = data.get("features", {})
            self._baseline_loaded = True
            logger.info(
                f"Drift baseline loaded: {len(self._baseline)} features "
                f"from {self._baseline_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load baseline.json: {e}")

    def update(self, feature_vector: dict):
        """
        Add a new feature vector to the rolling buffers.
        Call this on every /predict call.

        Args:
            feature_vector: dict with keys matching FEATURE_COLS
        """
        for col in FEATURE_COLS:
            val = feature_vector.get(col)
            if val is not None and math.isfinite(val):
                self._buffers[col].append(float(val))

    def compute_drift_scores(self) -> Dict[str, float]:
        """
        Compute KL-divergence drift score for each feature.

        KL(P_live || P_baseline) where distributions are approximated
        as Gaussians using mean and variance.

        Returns:
            dict mapping feature_name → drift_score (float)
            Updates Prometheus gauges as a side effect.
        """
        if not self._baseline_loaded:
            return {}

        scores = {}
        for col in FEATURE_COLS:
            buf = list(self._buffers[col])
            baseline = self._baseline.get(col)

            if not buf or baseline is None:
                scores[col] = 0.0
                continue

            if len(buf) < 30:
                # Not enough live data yet
                scores[col] = 0.0
                continue

            # Live distribution (Gaussian approximation)
            live_mean = float(np.mean(buf))
            live_var = float(np.var(buf)) + 1e-9  # avoid division by zero

            # Baseline distribution
            base_mean = baseline.get("mean", live_mean)
            base_var = baseline.get("variance", live_var) + 1e-9

            # KL divergence: KL(N(µ1,σ1²) || N(µ2,σ2²))
            # = log(σ2/σ1) + (σ1² + (µ1-µ2)²)/(2σ2²) - 1/2
            kl = (
                math.log(math.sqrt(base_var) / math.sqrt(live_var))
                + (live_var + (live_mean - base_mean) ** 2) / (2 * base_var)
                - 0.5
            )
            kl = max(0.0, round(kl, 6))  # KL is always >= 0
            scores[col] = kl

            # Update Prometheus gauge per feature
            FEATURE_DRIFT_SCORE.labels(feature=col).set(kl)

        # Overall drift = max across features
        overall = max(scores.values()) if scores else 0.0
        OVERALL_DRIFT_SCORE.set(overall)

        if overall > self._threshold:
            logger.warning(
                f"DATA DRIFT DETECTED: overall score={overall:.4f} "
                f"(threshold={self._threshold}). "
                f"Top drifted features: "
                f"{sorted(scores.items(), key=lambda x: -x[1])[:3]}"
            )

        return scores

    def is_drifting(self) -> bool:
        """True if current drift exceeds threshold."""
        scores = self.compute_drift_scores()
        return bool(scores and max(scores.values()) > self._threshold)

    def get_overall_score(self) -> float:
        """Return the current overall drift score without recomputing."""
        scores = self.compute_drift_scores()
        return max(scores.values()) if scores else 0.0

    def reset(self):
        """Clear all buffers (call on session end)."""
        for col in FEATURE_COLS:
            self._buffers[col].clear()


# ── Global singleton (shared across sessions) ─────────────────────────────────
# Each session gets its own DriftDetector instance in practice,
# but we also maintain a global one for the /metrics endpoint.

_global_detector: Optional[DriftDetector] = None


def get_global_detector() -> DriftDetector:
    """Lazy-init global drift detector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = DriftDetector()
    return _global_detector
