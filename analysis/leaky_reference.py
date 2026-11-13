"""
Analysis: how much did the v1 random split inflate the score?

Same features (main set), 15 s windows (closest to v1), v1 hyperparameters, threshold 0.5.
Only the split changes:
  random = v1 protocol: stratified random 80/20 over windows (same people in train and test)
  loso   = leave-one-participant-out (each person tested only while unseen)
Output: reports/leaky_vs_loso.json
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ddd_core.config import load_params
from ddd_core.windows import FEATURE_SETS
from pipeline.dataset import REPORTS_DIR, WINDOWS
from pipeline.evaluation import save_json, window_metrics


def v1_model(name):
    if name == "xgboost":  # v1 params.yaml values
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
            ),
        ]
    )


def main():
    params = load_params()
    ref = params["leaky_reference"]
    feats = FEATURE_SETS[params["features"]["main_set"]]
    df = pd.read_parquet(WINDOWS)
    df = df[df["window_s"] == ref["window_length_s"]].reset_index(drop=True)
    X, y = df[feats], df["label"].to_numpy()

    out = {"window_s": ref["window_length_s"], "features": params["features"]["main_set"], "threshold": 0.5}
    for name in ["xgboost", "svm"]:
        tr, te = train_test_split(
            np.arange(len(df)), test_size=ref["test_size"], stratify=y, random_state=ref["random_state"]
        )
        m = v1_model(name).fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        random_split = window_metrics(y[te], s, (s >= 0.5).astype(int))

        scores, preds, labels = [], [], []
        for p in sorted(df["participant"].unique()):
            tr_m, te_m = df["participant"] != p, df["participant"] == p
            m = v1_model(name).fit(X[tr_m], y[tr_m])
            s = m.predict_proba(X[te_m])[:, 1]
            scores.append(s)
            preds.append((s >= 0.5).astype(int))
            labels.append(y[te_m])
        loso = window_metrics(np.concatenate(labels), np.concatenate(scores), np.concatenate(preds))

        out[name] = {
            "random_split": random_split,
            "loso": loso,
            "f1_inflation": round(random_split["f1"] - loso["f1"], 4),
        }
        print(
            f"{name}: random-split F1 {random_split['f1']:.3f} AUC {random_split['roc_auc']:.3f} | "
            f"LOSO F1 {loso['f1']:.3f} AUC {loso['roc_auc']:.3f}"
        )
    save_json(out, REPORTS_DIR / "leaky_vs_loso.json")


if __name__ == "__main__":
    main()
