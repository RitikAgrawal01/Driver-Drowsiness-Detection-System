"""
End-to-end run of the REAL stage scripts on a tiny synthetic dataset:
frame_features -> window_features -> train_evaluate (nested LOSO) -> gate -> train_final,
logging to a throw-away file-based MLflow. Checks the outputs fit together and that the
registered @champion loads and predicts.
"""

import copy
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from ddd_core.geometry import USED_LANDMARKS
from ddd_core.windows import FEATURE_SETS
from tests.conftest import face_points

PARTICIPANTS, SECONDS, FPS = 6, 40, 10


def write_landmarks(dirpath, rng, participants):
    dirpath.mkdir(parents=True, exist_ok=True)
    for p in participants:
        for label in (0, 1):
            closed_prob = 0.25 if label else 0.03  # drowsy videos close the eyes far more often
            rows = []
            for i in range(SECONDS * FPS):
                ear = 0.1 if rng.random() < closed_prob else 0.3 + rng.normal(0, 0.01)
                pts = face_points(ear=ear)
                row = {
                    "video": f"{'drowsy' if label else 'alert'}{p}",
                    "participant": p,
                    "label": label,
                    "t": round(i / FPS, 3),
                    "width": 640,
                    "height": 480,
                    "face": 1,
                }
                for idx in USED_LANDMARKS:
                    row[f"x{idx}"], row[f"y{idx}"] = pts[idx]
                rows.append(row)
            pd.DataFrame(rows).to_csv(dirpath / f"{rows[0]['video']}.csv", index=False)


@pytest.fixture
def mini_pipeline(tmp_path, monkeypatch, params):
    import pipeline.frame_features as ff
    import pipeline.gate as gate
    import pipeline.train_evaluate as te
    import pipeline.train_final as tf
    import pipeline.window_features as wf

    p = copy.deepcopy(params)
    p["features"]["window_lengths_s"] = [10, 20]
    p["cv"].update(inner_splits=3, n_jobs=1)
    p["models"] = {
        "logreg": {"C": [1.0]},
        "svm": {"C": [1.0], "gamma": [0.1]},
        "xgboost": {
            **p["models"]["xgboost"],
            "max_depth": [2],
            "min_child_weight": [1],
            "max_estimators": 40,
            "early_stopping_rounds": 10,
        },
    }
    p["mlflow"] = {"experiment": "test_exp", "model_name": "test_model", "alias": "champion"}
    p["data"] = {"folds": [1, 2], "sealed_test_fold": 4}

    import pipeline.dataset as ds

    data, reports = tmp_path / "data", tmp_path / "reports"
    reports.mkdir(parents=True)
    rng = np.random.default_rng(0)
    write_landmarks(data / "landmarks" / "fold1", rng, range(1, 4))  # two folds, as in the real layout
    write_landmarks(data / "landmarks" / "fold2", rng, range(4, PARTICIPANTS + 1))
    monkeypatch.setattr(ds, "load_params", lambda: p)
    paths = {
        "LANDMARKS_DIR": data / "landmarks",
        "FRAME_FEATURES": data / "frames.parquet",
        "WINDOWS": data / "windows.parquet",
        "REPORTS_DIR": reports,
        "DATA_DIR": data,
    }
    for mod in (ff, wf, te, gate, tf):
        for name, value in paths.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)
        if hasattr(mod, "load_params"):
            monkeypatch.setattr(mod, "load_params", lambda: p)

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield {"ff": ff, "wf": wf, "te": te, "gate": gate, "tf": tf, "reports": reports, "data": data, "params": p}
    mlflow.set_tracking_uri(None)


def test_stages_end_to_end(mini_pipeline):
    m = mini_pipeline
    m["ff"].main()
    m["wf"].main()
    windows = pd.read_parquet(m["data"] / "windows.parquet")
    assert set(windows["window_s"]) == {10, 20} and windows["participant"].nunique() == PARTICIPANTS

    m["te"].main()
    metrics = json.loads((m["reports"] / "loso_metrics.json").read_text())
    assert "rule__perclos_only" in metrics and "xgboost__v2_12_no_pose" in metrics
    assert metrics["rule__perclos_only"]["pooled"]["roc_auc"] > 0.9  # PERCLOS separates the synthetic data
    choices = json.loads((m["reports"] / "loso_choices.json").read_text())
    assert {c["test_participant"] for c in choices} == set(range(1, PARTICIPANTS + 1))

    m["gate"].main()
    gate = json.loads((m["reports"] / "gate.json").read_text())
    assert gate["passed"] and gate["deploy"]["model"] in {"rule", "logreg", "svm", "xgboost"}

    m["tf"].main()
    final = json.loads((m["reports"] / "final_model.json").read_text())
    assert final["version"] == 1 and final["alias"] == "champion"
    model = mlflow.pyfunc.load_model("models:/test_model@champion")
    out = model.predict(windows[windows["window_s"] == final["window_s"]].head(3))
    assert list(out.columns) == ["score", "drowsy"] and len(out) == 3
    baseline = json.loads((m["data"] / "drift_baseline.json").read_text())
    assert set(baseline) == set(FEATURE_SETS[m["params"]["features"]["main_set"]])  # all 12 inputs monitored
