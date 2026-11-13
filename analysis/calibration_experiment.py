"""
EXPLORATORY: can per-driver normalisation and monotonic constraints make ML beat the PERCLOS rule?

These variants were designed after looking at Fold 1, so Fold 1 results are hints only. People
from later folds (reported separately) had never been examined: they are the first honest check.
Neither is the final test — that is the sealed fold.

Ideas tested (same nested leave-one-participant-out protocol as pipeline/train_evaluate.py):
  * centering: every feature minus the driver's own median over all of their windows (no labels).
    Removes person/camera offsets (eye shape, phone angle). Caveat: at test time it uses the new
    driver's unlabelled history, and here that history is exactly half alert / half drowsy.
    (Calibrating on the alert video alone is NOT used: 4 of 12 participants filmed their two
    videos with different camera setups, so "difference from the alert video" would partly
    measure the camera change, not drowsiness.)
  * monotonic XGBoost: more PERCLOS / longer blinks / more long closures can only raise the score,
    higher EAR can only lower it, so the model cannot learn "backwards" person-specific rules.

Output: reports/exploratory_calibration.json, reports/EXPLORATORY_CALIBRATION.md
"""

from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ddd_core.config import load_params
from ddd_core.windows import FEATURE_SETS
from pipeline.dataset import REPORTS_DIR, WINDOWS
from pipeline.evaluation import save_json, summarize
from pipeline.models import best_f1_threshold

MAIN = FEATURE_SETS["v2_12"]
NO_POSE = FEATURE_SETS["v2_12_no_pose"]


def center_per_driver(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Subtract each participant's median (per window length) from every feature. Uses no labels."""
    out = df.copy()
    med = df.groupby(["participant", "window_s"])[feats].transform("median")
    out[feats] = df[feats] - med
    return out


def monotone_tuple(feats: list, directions: dict) -> tuple:
    return tuple(int(directions.get(f, 0)) for f in feats)


def configs(model: str, mp: dict) -> list[dict]:
    if model == "rule":
        return [{}]
    if model == "logreg":
        return [{"C": c} for c in mp["logreg"]["C"]]
    if model == "svm":
        return [{"C": c, "gamma": g} for c, g in product(mp["svm"]["C"], mp["svm"]["gamma"])]
    x = mp["xgboost"]
    return [{"max_depth": d, "min_child_weight": m} for d, m in product(x["max_depth"], x["min_child_weight"])]


def build(model: str, cfg: dict, mp: dict, feats: list, mono: dict, n_estimators=None, early_stop=False):
    if model == "logreg":
        return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=cfg["C"], max_iter=5000))])
    if model == "svm":
        return Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"]))])
    x = mp["xgboost"]
    return XGBClassifier(
        n_estimators=x["max_estimators"] if early_stop else n_estimators,
        early_stopping_rounds=x["early_stopping_rounds"] if early_stop else None,
        learning_rate=x["learning_rate"],
        subsample=x["subsample"],
        colsample_bytree=x["colsample_bytree"],
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
        monotone_constraints=monotone_tuple(feats, mono) if model == "xgb_mono" else None,
        **cfg,
    )


def score(model, est, X):
    if model == "rule":
        return X["perclos"].to_numpy()
    if model == "svm":
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def select(model, data, feats, mp, mono, n_splits):
    best = None
    for L, df in data.items():
        X, y, g = df[feats], df["label"].to_numpy(), df["participant"].to_numpy()
        folds = list(GroupKFold(n_splits=n_splits).split(X, y, g))
        for cfg in configs(model, mp):
            oof, aucs, iters = np.zeros(len(df)), [], []
            for tr, va in folds:
                if model == "rule":
                    s = score(model, None, X.iloc[va])
                elif model.startswith("xgb"):
                    est = build(model, cfg, mp, feats, mono, early_stop=True)
                    est.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
                    iters.append(est.best_iteration + 1)
                    s = score(model, est, X.iloc[va])
                else:
                    s = score(model, build(model, cfg, mp, feats, mono).fit(X.iloc[tr], y[tr]), X.iloc[va])
                oof[va] = s
                aucs.append(roc_auc_score(y[va], s))
            cand = {
                "window_s": int(L),
                "config": cfg,
                "inner_auc": float(np.mean(aucs)),
                "oof": oof,
                "n_estimators": int(round(np.mean(iters))) if iters else None,
            }
            if best is None or cand["inner_auc"] > best["inner_auc"]:
                best = cand
    best["threshold"] = best_f1_threshold(data[best["window_s"]]["label"].to_numpy(), best.pop("oof"))
    return best


def outer(name, model, feats, by_len, test_p, mp, mono, n_splits):
    train = {L: d[d["participant"] != test_p] for L, d in by_len.items()}
    c = select(model, train, feats, mp, mono, n_splits)
    tr = train[c["window_s"]]
    est = None
    if model != "rule":
        est = build(model, c["config"], mp, feats, mono, n_estimators=c["n_estimators"])
        est.fit(tr[feats], tr["label"].to_numpy())
    te = by_len[c["window_s"]][by_len[c["window_s"]]["participant"] == test_p]
    s = score(model, est, te[feats])
    out = te[["video", "participant", "label", "window_s"]].copy()
    out["score"], out["pred"], out["run"] = s, (s >= c["threshold"]).astype(int), name
    return out


def main():
    params = load_params()
    mp, cv = params["models"], params["cv"]
    mono = params["exploratory"]["monotone"]
    raw = pd.read_parquet(WINDOWS)
    centered = center_per_driver(raw, MAIN)
    data = {"raw": raw, "centered": centered}

    variants = [
        ("rule (raw)", "rule", "raw", ["perclos"]),
        ("rule (centered)", "rule", "centered", ["perclos"]),
        ("logreg (centered)", "logreg", "centered", MAIN),
        ("logreg (centered, no pose)", "logreg", "centered", NO_POSE),
        ("svm (centered)", "svm", "centered", MAIN),
        ("svm (centered, no pose)", "svm", "centered", NO_POSE),
        ("xgb monotonic (raw)", "xgb_mono", "raw", MAIN),
        ("xgb monotonic (raw, no pose)", "xgb_mono", "raw", NO_POSE),
        ("xgb monotonic (centered)", "xgb_mono", "centered", MAIN),
        ("xgb monotonic (centered, no pose)", "xgb_mono", "centered", NO_POSE),
    ]
    by_len = {k: {int(L): d.reset_index(drop=True) for L, d in df.groupby("window_s")} for k, df in data.items()}
    participants = sorted(raw["participant"].unique())
    tasks = [(n, m, f, by_len[v], p) for n, m, v, f in variants for p in participants]
    parts = Parallel(n_jobs=cv["n_jobs"], verbose=2)(
        delayed(outer)(n, m, f, bl, p, mp, mono, cv["inner_splits"]) for n, m, f, bl, p in tasks
    )
    preds = pd.concat(parts, ignore_index=True)
    fold_of = raw.groupby("participant")["fold"].first()
    preds["fold"] = preds["participant"].map(fold_of)

    groups = {
        "new folds (never examined before)": preds[preds["fold"] != 1],
        "fold 1 (already examined many times)": preds[preds["fold"] == 1],
        "all participants": preds,
    }
    out, tables = {}, {}
    for group, gp in groups.items():
        if gp.empty:
            continue
        results, per_person = {}, {}
        for name, *_ in variants:
            summary, pp = summarize(gp[gp["run"] == name])
            results[name] = summary
            per_person[name] = pp.set_index("participant")["roc_auc"]
        base = per_person["rule (raw)"]
        for name in results:
            diff = per_person[name] - base
            nonzero = diff[diff.abs() > 1e-12]
            results[name]["vs_rule_raw"] = {
                "better_for_people": f"{int((diff > 1e-12).sum())}/{len(diff)}",
                "mean_auc_diff": round(float(diff.mean()), 4),
                "wilcoxon_p": round(float(wilcoxon(nonzero).pvalue), 4) if len(nonzero) >= 5 else None,
                "worst_person_auc": round(float(per_person[name].min()), 3),
            }
        out[group] = results
        tables[group] = pd.DataFrame(per_person).round(3)
    save_json(
        {"results": out, "per_person_auc": {g: t.to_dict() for g, t in tables.items()}},
        REPORTS_DIR / "exploratory_calibration.json",
    )
    write_md(out, tables)


def write_md(out: dict, tables: dict):
    L = [
        "# EXPLORATORY: per-driver centering + monotonic XGBoost",
        "",
        "The variants were designed after looking at Fold 1, so Fold 1 numbers are hints only. People from",
        "later folds had never been examined: their table is the first honest check of these ideas.",
        "Protocol: nested leave-one-participant-out over all participants, as in the main pipeline.",
        "Centering = feature minus the driver's own median over all their windows (no labels).",
    ]
    for group, results in out.items():
        L += [
            "",
            f"## {group}",
            "",
            "| Variant | ROC-AUC / person | Pooled ROC-AUC | Pooled F1 | Precision | Recall | Specificity | "
            "Better than rule for | Mean diff | Wilcoxon p | Worst person |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, r in results.items():
            p, mu, sd, v = r["pooled"], r["per_participant_mean"], r["per_participant_std"], r["vs_rule_raw"]
            L.append(
                f"| {name} | {mu['roc_auc']:.3f} ± {sd['roc_auc']:.3f} | {p['roc_auc']:.3f} | {p['f1']:.3f} | "
                f"{p['precision']:.3f} | {p['recall']:.3f} | {p['specificity']:.3f} | {v['better_for_people']} | "
                f"{v['mean_auc_diff']:+.3f} | {v['wilcoxon_p']} | {v['worst_person_auc']:.2f} |"
            )
    pp = tables["all participants"]
    cols = list(pp.columns)
    L += [
        "",
        "## ROC-AUC per held-out person",
        "",
        "| Person | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for person, row in pp.iterrows():
        L.append(f"| {person} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    L.append("")
    (REPORTS_DIR / "EXPLORATORY_CALIBRATION.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
