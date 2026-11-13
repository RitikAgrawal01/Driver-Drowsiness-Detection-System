"""Model zoo + the grouped selection loop shared by the LOSO evaluation and the final fit."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ["rule", "logreg", "svm", "xgboost"]


def configs(model: str, mp: dict) -> list[dict]:
    if model == "rule":
        return [{}]
    if model == "logreg":
        return [{"C": c} for c in mp["logreg"]["C"]]
    if model == "svm":
        return [{"C": c, "gamma": g} for c, g in product(mp["svm"]["C"], mp["svm"]["gamma"])]
    if model == "xgboost":
        x = mp["xgboost"]
        return [{"max_depth": d, "min_child_weight": m} for d, m in product(x["max_depth"], x["min_child_weight"])]
    raise ValueError(model)


def build(model: str, cfg: dict, mp: dict, n_estimators: int | None = None, early_stop: bool = False):
    if model == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=cfg["C"], max_iter=5000))])
    if model == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"]))])
    if model == "xgboost":
        x = mp["xgboost"]
        return XGBClassifier(
            n_estimators=x["max_estimators"] if early_stop else n_estimators,
            early_stopping_rounds=x["early_stopping_rounds"] if early_stop else None,
            learning_rate=x["learning_rate"], subsample=x["subsample"],
            colsample_bytree=x["colsample_bytree"], eval_metric="logloss",
            random_state=42, n_jobs=1, verbosity=0, **cfg)
    raise ValueError(model)


def score(model: str, est, X: pd.DataFrame) -> np.ndarray:
    """Continuous drowsiness score (higher = more drowsy)."""
    if model == "rule":
        return X["perclos"].to_numpy()
    if model == "svm":
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def select(model: str, data: dict, features: list, mp: dict, n_splits: int) -> dict:
    """Pick window length + config by GroupKFold over participants (the validation step).

    data: {window_s: DataFrame of training windows}. Each validation fold holds whole
    participants (both their videos), so validation never shares a person with training.
    Returns window_s, config, mean validation ROC-AUC, out-of-fold F1 threshold and
    (XGBoost) the early-stopped number of trees.
    """
    best = None
    for L, df in data.items():
        X, y, g = df[features], df["label"].to_numpy(), df["participant"].to_numpy()
        folds = list(GroupKFold(n_splits=n_splits).split(X, y, g))
        for cfg in configs(model, mp):
            oof, aucs, iters = np.zeros(len(df)), [], []
            for tr, va in folds:
                if model == "rule":
                    s = score(model, None, X.iloc[va])
                elif model == "xgboost":
                    est = build(model, cfg, mp, early_stop=True)
                    est.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
                    iters.append(est.best_iteration + 1)
                    s = score(model, est, X.iloc[va])
                else:
                    est = build(model, cfg, mp).fit(X.iloc[tr], y[tr])
                    s = score(model, est, X.iloc[va])
                oof[va] = s
                aucs.append(roc_auc_score(y[va], s))
            cand = {"window_s": int(L), "config": cfg, "inner_auc": float(np.mean(aucs)),
                    "oof": oof, "n_estimators": int(round(np.mean(iters))) if iters else None}
            if best is None or cand["inner_auc"] > best["inner_auc"]:
                best = cand
    df = data[best["window_s"]]
    best["threshold"] = best_f1_threshold(df["label"].to_numpy(), best.pop("oof"))
    return best


def fit_final(model: str, choice: dict, df: pd.DataFrame, features: list, mp: dict):
    if model == "rule":
        return None
    est = build(model, choice["config"], mp, n_estimators=choice["n_estimators"])
    return est.fit(df[features], df["label"].to_numpy())
