"""Population Stability Index (PSI) drift per feature against the training baseline.

Baseline = quantile bin edges + training proportions per feature, written by the train_final
stage (data/drift_baseline.json, also logged with the champion in MLflow).
PSI = sum((live - base) * ln(live / base)); < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift.
Unlike v1's Gaussian KL, PSI makes no normality assumption (PERCLOS piles up at 0, angles wrap).
"""

import numpy as np

EPS = 1e-4


def psi(live: np.ndarray, edges: list, base_props: list) -> float:
    live = np.asarray(live, dtype=float)
    live = live[~np.isnan(live)]
    if live.size == 0:
        return float("nan")
    counts, _ = np.histogram(live, bins=np.asarray(edges, dtype=float))
    p = np.clip(counts / counts.sum(), EPS, None)
    q = np.clip(np.asarray(base_props, dtype=float), EPS, None)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_all(live_rows: list[dict], baseline: dict) -> dict:
    """PSI for every feature in the baseline, given live window feature dicts."""
    out = {}
    for feat, b in baseline.items():
        out[feat] = psi(np.array([r[feat] for r in live_rows if feat in r]), b["edges"], b["proportions"])
    return out
