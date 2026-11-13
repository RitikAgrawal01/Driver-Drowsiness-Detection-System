"""
Stage 4: train and compare models with nested leave-one-participant-out (LOSO).

Outer loop: each participant (all folds in params.yaml data.folds) is the TEST set once
(both of their videos).
Inner loop (validation): on the other 11, GroupKFold by participant chooses the window length
(15/30/60 s), hyperparameters, XGBoost's number of trees (early stopping) and the decision
threshold (max F1 on out-of-fold scores). That setup is refit on the 11 and scored once on
the held-out person, who was never used for any choice.

Runs: PERCLOS rule (baseline), logistic regression (baseline), SVM (RBF), XGBoost,
      x feature sets v1_9_fixed, v2_12 (main), v2_12_no_pose (head-pose shortcut ablation).
Outputs: data/loso_predictions.parquet, reports/loso_metrics.json,
         reports/loso_per_participant.csv, reports/loso_choices.json; one MLflow run each.
"""

import time

import mlflow
import pandas as pd
from joblib import Parallel, delayed

from ddd_core.config import load_params
from ddd_core.windows import FEATURE_SETS
from pipeline.dataset import DATA_DIR, REPORTS_DIR, WINDOWS
from pipeline.evaluation import dvc_lock_hash, flat, git_commit, mlflow_setup, save_json, summarize
from pipeline.models import fit_final, score, select


def outer_fold(model, fs_name, features, test_p, by_len, mp, n_splits):
    train = {L: d[d["participant"] != test_p] for L, d in by_len.items()}
    choice = select(model, train, features, mp, n_splits)
    L = choice["window_s"]
    est = fit_final(model, choice, train[L], features, mp)
    test = by_len[L][by_len[L]["participant"] == test_p]
    s = score(model, est, test[features])
    out = test[["video", "participant", "label", "window_s", "start_s"]].copy()
    out["score"], out["pred"] = s, (s >= choice["threshold"]).astype(int)
    out["model"], out["feature_set"] = model, fs_name
    return out, {"model": model, "feature_set": fs_name, "test_participant": int(test_p), **choice}


def main():
    params = load_params()
    mp, cv = params["models"], params["cv"]
    windows = pd.read_parquet(WINDOWS)
    by_len = {int(L): d.reset_index(drop=True) for L, d in windows.groupby("window_s")}
    participants = sorted(windows["participant"].unique())

    runs = [("rule", "perclos_only", ["perclos"])]
    runs += [(m, fs, FEATURE_SETS[fs]) for m in ["logreg", "svm", "xgboost"] for fs in FEATURE_SETS]

    t0 = time.time()
    tasks = [(m, fs, feats, p) for m, fs, feats in runs for p in participants]
    results = Parallel(n_jobs=cv["n_jobs"], verbose=5)(
        delayed(outer_fold)(m, fs, feats, p, by_len, mp, cv["inner_splits"]) for m, fs, feats, p in tasks
    )
    elapsed = time.time() - t0
    print(f"nested LOSO done in {elapsed:.0f} s")

    preds = pd.concat([r[0] for r in results], ignore_index=True)
    choices = [r[1] for r in results]
    preds.to_parquet(DATA_DIR / "loso_predictions.parquet", index=False)
    save_json(choices, REPORTS_DIR / "loso_choices.json")

    mlflow_setup(params["mlflow"]["experiment"])
    metrics, per_part_all = {}, []
    for m, fs, feats in runs:
        key = f"{m}__{fs}"
        summary, per_part = summarize(preds[(preds["model"] == m) & (preds["feature_set"] == fs)])
        chosen = [c for c in choices if c["model"] == m and c["feature_set"] == fs]
        counts = pd.Series([c["window_s"] for c in chosen]).value_counts()
        summary["chosen_window_s"] = {str(int(k)): int(v) for k, v in counts.items()}
        metrics[key] = summary
        per_part.insert(0, "run", key)
        per_part_all.append(per_part)

        with mlflow.start_run(run_name=key):
            mlflow.set_tags(
                {
                    "stage": "train_evaluate",
                    "git_commit": git_commit(),
                    "protocol": "nested LOSO",
                    "model": m,
                    "feature_set": fs,
                    "dvc_windows_md5": dvc_lock_hash("data/windows.parquet"),
                }
            )
            mlflow.log_params(
                {
                    "model": m,
                    "feature_set": fs,
                    "features": ",".join(feats),
                    "outer": "leave-one-participant-out (12)",
                    "inner": f"GroupKFold({cv['inner_splits']}) by participant",
                    "window_candidates_s": str(params["features"]["window_lengths_s"]),
                    "target_fps": params["extract"]["target_fps"],
                }
            )
            mlflow.log_metrics(
                flat(
                    {
                        "pooled": summary["pooled"],
                        "pp_mean": summary["per_participant_mean"],
                        "pp_std": summary["per_participant_std"],
                    }
                )
            )
            mlflow.log_metric("video_level_accuracy", summary["video_level_accuracy"])
            mlflow.log_dict(chosen, "fold_choices.json")
            mlflow.log_text(per_part.to_csv(index=False), "per_participant.csv")
        print(
            f"{key:28s} pooled F1 {summary['pooled']['f1']:.3f}  AUC {summary['pooled']['roc_auc']:.3f}  "
            f"per-participant AUC {summary['per_participant_mean']['roc_auc']:.3f} "
            f"± {summary['per_participant_std']['roc_auc']:.3f}  videos {summary['video_level_correct']}"
        )

    # Deterministic content only (no timings), so an identical re-run gives an identical file/md5.
    metrics["_meta"] = {"participants": [int(p) for p in participants]}
    save_json(metrics, REPORTS_DIR / "loso_metrics.json")
    pd.concat(per_part_all).to_csv(REPORTS_DIR / "loso_per_participant.csv", index=False)


if __name__ == "__main__":
    main()
