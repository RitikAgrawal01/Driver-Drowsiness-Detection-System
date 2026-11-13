"""
Stage 6: fit the gated candidate on ALL 12 participants and register it as `@champion`.
Runs only if the gate stage passed (DVC stops the pipeline when gate.py exits 1).

Window length, hyperparameters, number of trees and threshold are chosen with GroupKFold over
participants (same procedure as the inner LOSO loop). The performance we report for this model
is the nested LOSO estimate, never a score on data it was trained on.
Outputs: reports/final_model.json, data/drift_baseline.json; MLflow model with alias.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from ddd_core.config import REPO_ROOT, load_params
from ddd_core.windows import FEATURE_SETS
from pipeline.dataset import DATA_DIR, REPORTS_DIR, WINDOWS
from pipeline.evaluation import dvc_lock_hash, file_md5, flat, git_commit, mlflow_setup, save_json
from pipeline.models import fit_final, select
from pipeline.pyfunc_model import DrowsinessModel


def drift_baseline(train: pd.DataFrame, feats: list, bins: int) -> dict:
    """Per-feature quantile bin edges + training proportions, for PSI drift in the backend."""
    out = {}
    for c in feats:
        x = train[c].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(x, bins=edges)
        out[c] = {
            "edges": [float(e) if np.isfinite(e) else (1e308 if e > 0 else -1e308) for e in edges],
            "proportions": (counts / counts.sum()).round(6).tolist(),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
    return out


def main():
    params = load_params()
    gate = json.loads((REPORTS_DIR / "gate.json").read_text())
    if not gate["passed"] or not gate.get("deploy"):
        raise SystemExit("Gate did not pass; refusing to promote.")
    deploy = gate["deploy"]
    metrics_path = REPORTS_DIR / "loso_metrics.json"
    loso = json.loads(metrics_path.read_text())[deploy["run_key"]]

    model, main_set = deploy["model"], params["features"]["main_set"]
    feats = ["perclos"] if model == "rule" else FEATURE_SETS[deploy["feature_set"]]
    windows = pd.read_parquet(WINDOWS)
    by_len = {int(L): d.reset_index(drop=True) for L, d in windows.groupby("window_s")}
    choice = select(model, by_len, feats, params["models"], params["cv"]["inner_splits"])
    train = by_len[choice["window_s"]]
    est = fit_final(model, choice, train, feats, params["models"])

    # Drift is monitored on ALL model inputs the backend computes, whatever the champion uses.
    baseline = drift_baseline(train, FEATURE_SETS[main_set], params["drift"]["bins"])
    save_json(baseline, DATA_DIR / "drift_baseline.json")

    ml = params["mlflow"]
    mlflow_setup(ml["experiment"])
    with mlflow.start_run(run_name=f"final__{model}") as run:
        mlflow.set_tags(
            {
                "stage": "train_final",
                "git_commit": git_commit(),
                "gate": "passed",
                "deploy_reason": deploy["reason"],
                "ml_gate_passed": str(gate["ml_gate_passed"]),
                "dvc_windows_md5": dvc_lock_hash("data/windows.parquet"),
                "loso_metrics_md5": file_md5(metrics_path),
                "loso_run_key": deploy["run_key"],
            }
        )
        mlflow.log_params(
            {
                "model": model,
                "feature_set": deploy["feature_set"],
                "features": ",".join(feats),
                "window_s": choice["window_s"],
                "threshold": round(choice["threshold"], 6),
                "n_estimators": choice["n_estimators"],
                "train_participants": 12,
                "train_windows": len(train),
                **{f"cfg_{k}": v for k, v in choice["config"].items()},
            }
        )
        mlflow.log_metrics(flat({"loso_pooled": loso["pooled"], "loso_pp_mean": loso["per_participant_mean"]}))
        mlflow.log_dict(baseline, "drift_baseline.json")
        example = train[feats].head(5)
        wrapped = DrowsinessModel(est, model, feats, choice["threshold"], choice["window_s"])
        info = mlflow.pyfunc.log_model(
            "model",
            python_model=wrapped,
            code_path=[str(REPO_ROOT / "pipeline")],
            input_example=example,
            signature=infer_signature(example, wrapped.predict(None, example)),
            metadata={
                "window_s": choice["window_s"],
                "threshold": choice["threshold"],
                "features": feats,
                "model_type": model,
                "target_fps": params["extract"]["target_fps"],
            },
            registered_model_name=ml["model_name"],
        )
        client = MlflowClient()
        # mlflow 2.11's ModelInfo has no registered_model_version: look the version up by run id
        versions = client.search_model_versions(f"name='{ml['model_name']}' and run_id='{run.info.run_id}'")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(ml["model_name"], ml["alias"], str(version))
        print(f"registered {ml['model_name']} v{version} as @{ml['alias']} ({info.model_uri})")

    result = {
        "model": model,
        "feature_set": deploy["feature_set"],
        "deploy_reason": deploy["reason"],
        "ml_gate_passed": gate["ml_gate_passed"],
        "window_s": choice["window_s"],
        "config": choice["config"],
        "n_estimators": choice["n_estimators"],
        "threshold": round(choice["threshold"], 6),
        "selection_cv_auc": round(choice["inner_auc"], 4),
        "train_windows": len(train),
        "mlflow_run_id": run.info.run_id,
        "registered_model": ml["model_name"],
        "version": int(version),
        "alias": ml["alias"],
        "reported_performance": "nested LOSO (reports/loso_metrics.json)",
    }
    save_json(result, REPORTS_DIR / "final_model.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
