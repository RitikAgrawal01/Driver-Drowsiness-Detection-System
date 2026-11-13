"""Exploratory calibration helpers: per-driver centering and monotonic constraints."""

import numpy as np
import pandas as pd

from analysis.calibration_experiment import build, center_per_driver, monotone_tuple


def test_centering_removes_each_drivers_offset_without_labels():
    df = pd.DataFrame(
        {
            "participant": [1, 1, 1, 2, 2, 2],
            "window_s": 15,
            "ear_mean": [0.30, 0.32, 0.34, 0.20, 0.22, 0.24],
            "label": [0, 1, 1, 0, 0, 1],
        }
    )
    out = center_per_driver(df, ["ear_mean"])
    assert np.allclose(out["ear_mean"], [-0.02, 0, 0.02, -0.02, 0, 0.02])
    assert out["label"].tolist() == df["label"].tolist()  # labels untouched
    assert np.isclose(df["ear_mean"].iloc[0], 0.30)  # input not modified


def test_monotone_tuple_follows_feature_order():
    assert monotone_tuple(["ear_mean", "mar_max", "perclos"], {"perclos": 1, "ear_mean": -1}) == (-1, 0, 1)


def test_monotonic_xgboost_cannot_learn_a_backwards_relation(params):
    rng = np.random.default_rng(0)
    perclos = rng.random(400)
    y = (perclos < 0.5).astype(int)  # deliberately "backwards" data
    X = pd.DataFrame({"perclos": perclos})
    model = build(
        "xgb_mono",
        {"max_depth": 2, "min_child_weight": 1},
        params["models"],
        ["perclos"],
        {"perclos": 1},
        n_estimators=50,
    ).fit(X, y)
    s = model.predict_proba(pd.DataFrame({"perclos": np.linspace(0, 1, 50)}))[:, 1]
    assert np.all(np.diff(s) >= -1e-9)  # score never decreases as PERCLOS rises
