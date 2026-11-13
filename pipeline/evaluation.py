"""Metrics and MLflow helpers shared by the pipeline stages."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ddd_core.config import REPO_ROOT

METRIC_KEYS = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]


def window_metrics(y: np.ndarray, s: np.ndarray, pred: np.ndarray) -> dict:
    """Drowsy = positive class."""
    m = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "specificity": recall_score(1 - y, 1 - pred, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, s)
        m["pr_auc"] = average_precision_score(y, s)
    return {k: round(float(v), 4) for k, v in m.items()}


def summarize(pred_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pooled metrics, per-participant metrics (mean ± std) and video-level majority vote."""
    y, s = pred_df["label"].to_numpy(), pred_df["score"].to_numpy()
    p = pred_df["pred"].to_numpy().astype(int)
    pooled = window_metrics(y, s, p)

    rows = []
    for part, g in pred_df.groupby("participant"):
        r = window_metrics(g["label"].to_numpy(), g["score"].to_numpy(), g["pred"].to_numpy().astype(int))
        r.update(participant=int(part), window_s=int(g["window_s"].iloc[0]), n_windows=len(g))
        rows.append(r)
    per_part = pd.DataFrame(rows)

    vids = pred_df.groupby("video").agg(label=("label", "first"), drowsy_share=("pred", "mean"))
    vids["video_pred"] = (vids["drowsy_share"] >= 0.5).astype(int)
    correct = int((vids["video_pred"] == vids["label"]).sum())

    summary = {
        "pooled": pooled,
        "per_participant_mean": {k: round(float(per_part[k].mean()), 4) for k in METRIC_KEYS},
        "per_participant_std": {k: round(float(per_part[k].std(ddof=0)), 4) for k in METRIC_KEYS},
        "worst_participant_recall": round(float(per_part["recall"].min()), 4),
        "video_level_correct": f"{correct}/{len(vids)}",
        "video_level_accuracy": round(correct / len(vids), 4),
        "n_windows": int(len(pred_df)),
    }
    return summary, per_part


def mlflow_setup(experiment: str) -> None:
    """Log to the MLflow server named by MLFLOW_TRACKING_URI (infra setting, from .env / run script)."""
    import mlflow

    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set (e.g. http://localhost:5000). Refusing to log nowhere.")
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return r.stdout.strip() or "unknown"


def dvc_lock_hash(path: str) -> str:
    """md5 that dvc.lock records for an output (real data lineage for MLflow tags)."""
    import yaml

    lock = REPO_ROOT / "dvc.lock"
    if not lock.exists():
        return "no_dvc_lock"
    stages = yaml.safe_load(lock.read_text()).get("stages", {})
    for st in stages.values():
        for o in st.get("outs", []):
            if o["path"] == path:
                return o.get("md5", "unknown")
    return "not_in_dvc_lock"


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}_"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
