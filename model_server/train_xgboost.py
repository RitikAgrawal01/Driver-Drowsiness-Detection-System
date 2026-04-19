"""
model_server/train_xgboost.py
──────────────────────────────
Stage 5 of the DVC pipeline.

Trains an XGBoost classifier on the drowsiness feature dataset.
Logs everything to MLflow:
  • Parameters    : all hyperparameters + dataset info
  • Metrics       : F1, accuracy, precision, recall, AUC-ROC, inference latency
  • Artifacts     : trained model, confusion matrix, feature importance plot
  • Tags          : git commit SHA, DVC data hash, dataset size

Also registers the model in the MLflow Model Registry under
MLFLOW_MODEL_NAME with stage 'Staging'.
The evaluate.py script later promotes the best model to 'Production'.

Can be run:
  • Standalone : python model_server/train_xgboost.py
  • By DVC     : dvc repro train_xgboost
"""

import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_xgboost")

# ── Config from environment / params.yaml ────────────────────────────────────
TRAIN_CSV = os.getenv("TRAIN_CSV", "data/processed/train.csv")
TEST_CSV = os.getenv("TEST_CSV", "data/processed/test.csv")
MODEL_OUTPUT = os.getenv("XGB_MODEL_OUTPUT", "models/xgboost_model.pkl")
METRICS_OUTPUT = os.getenv("XGB_METRICS_OUTPUT", "reports/xgboost_metrics.json")
PARAMS_FILE = os.getenv("PARAMS_FILE", "params.yaml")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "drowsiness_detection")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "drowsiness_classifier")

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]


def load_params() -> dict:
    """Load parameters from params.yaml."""
    if not os.path.isfile(PARAMS_FILE):
        logger.warning(f"params.yaml not found at {PARAMS_FILE}, using defaults")
        return {}
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Get current git commit SHA for MLflow tagging."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_dvc_data_hash(filepath: str) -> str:
    """Get DVC md5 hash of a tracked file for reproducibility tagging."""
    dvc_file = filepath + ".dvc"
    if os.path.isfile(dvc_file):
        with open(dvc_file) as f:
            content = yaml.safe_load(f)
        return content.get("outs", [{}])[0].get("md5", "unknown")
    return "not_dvc_tracked"


def load_data(train_csv: str, test_csv: str) -> tuple:
    """Load and validate train/test splits."""
    logger.info(f"Loading train data: {train_csv}")
    train_df = pd.read_csv(train_csv)

    logger.info(f"Loading test data: {test_csv}")
    test_df = pd.read_csv(test_csv)

    # Validate columns
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in train CSV: {missing}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values

    logger.info(
        f"Data loaded — train: {X_train.shape}, test: {X_test.shape}\n"
        f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}\n"
        f"  Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}"
    )
    return X_train, y_train, X_test, y_test, train_df, test_df


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str) -> str:
    """Plot and save a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Alert", "Drowsy"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, color="white")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, color="white")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black",
                fontsize=14, fontweight="bold",
            )

    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual", color="white", fontsize=12)
    ax.set_title(title, color="#06B6D4", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2A3A")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")
    return output_path


def plot_feature_importance(model: XGBClassifier, output_path: str) -> str:
    """Plot and save XGBoost feature importance."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    bars = ax.barh(
        [FEATURE_COLS[i] for i in sorted_idx],
        importance[sorted_idx],
        color="#06B6D4",
        edgecolor="#1E2A3A",
    )

    ax.set_xlabel("Feature Importance (gain)", color="white")
    ax.set_title("XGBoost Feature Importance", color="#06B6D4",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_facecolor("#0D1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2A3A")

    # Add value labels on bars
    for bar, val in zip(bars, importance[sorted_idx]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Feature importance saved: {output_path}")
    return output_path


def measure_inference_latency(model, X_test: np.ndarray, n_runs: int = 100) -> dict:
    """Measure single-sample inference latency (P50, P95, P99)."""
    latencies = []
    single_sample = X_test[0:1]  # one window

    # Warm up
    for _ in range(10):
        model.predict_proba(single_sample)

    # Measure
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict_proba(single_sample)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
    }


def train(params: dict) -> dict:
    """
    Main training function.
    Returns a dict with all metrics for DVC metrics tracking.
    """
    xgb_params = params.get("xgboost", {})
    feature_params = params.get("features", {})

    # ── Configure MLflow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, train_df, test_df = load_data(
        TRAIN_CSV, TEST_CSV
    )

    # ── Compute scale_pos_weight for class imbalance ──────────────────────────
    n_alert = int((y_train == 0).sum())
    n_drowsy = int((y_train == 1).sum())
    # Use param override or auto-compute
    scale_pos_weight = xgb_params.get("scale_pos_weight", 1)
    if scale_pos_weight == 1 and n_drowsy > 0:
        scale_pos_weight = round(n_alert / n_drowsy, 3)
        logger.info(f"Auto scale_pos_weight: {scale_pos_weight} ({n_alert} alert / {n_drowsy} drowsy)")

    # ── Start MLflow run ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost_training") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Log parameters ────────────────────────────────────────────────────
        # All XGBoost hyperparameters
        mlflow.log_params({
            "model_type": "XGBoost",
            "n_estimators": xgb_params.get("n_estimators", 200),
            "max_depth": xgb_params.get("max_depth", 6),
            "learning_rate": xgb_params.get("learning_rate", 0.1),
            "subsample": xgb_params.get("subsample", 0.8),
            "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
            "min_child_weight": xgb_params.get("min_child_weight", 1),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": xgb_params.get("eval_metric", "logloss"),
            # Feature engineering params
            "sliding_window_size": feature_params.get("sliding_window_size", 30),
            "ear_threshold": feature_params.get("ear_threshold", 0.25),
            "mar_threshold": feature_params.get("mar_threshold", 0.60),
            # Dataset info
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(FEATURE_COLS),
            "feature_names": ",".join(FEATURE_COLS),
            "train_csv": TRAIN_CSV,
        })

        # ── Tags for reproducibility ──────────────────────────────────────────
        git_sha = get_git_commit()
        data_hash = get_dvc_data_hash(TRAIN_CSV)
        mlflow.set_tags({
            "git_commit": git_sha,
            "dvc_data_hash": data_hash,
            "train_samples_alert": n_alert,
            "train_samples_drowsy": n_drowsy,
            "mlflow.note.content": (
                f"XGBoost drowsiness classifier\n"
                f"Git: {git_sha} | Data hash: {data_hash}"
            ),
        })

        # ── Build and train model ─────────────────────────────────────────────
        model = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            min_child_weight=xgb_params.get("min_child_weight", 1),
            scale_pos_weight=scale_pos_weight,
            eval_metric=xgb_params.get("eval_metric", "logloss"),
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,  # use all CPU cores
            verbosity=0,
        )

        logger.info("Training XGBoost model...")
        train_start = time.time()

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        train_time = time.time() - train_start
        logger.info(f"Training complete in {train_time:.1f}s")
        mlflow.log_metric("training_time_sec", round(train_time, 2))

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred, average="weighted")
        f1_drowsy = f1_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        auc_roc = roc_auc_score(y_test, y_proba)

        # Inference latency
        latency = measure_inference_latency(model, X_test)

        # ── Log all metrics ───────────────────────────────────────────────────
        metrics = {
            "f1_weighted": round(f1, 4),
            "f1_drowsy_class": round(f1_drowsy, 4),
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "auc_roc": round(auc_roc, 4),
            **latency,
            "meets_latency_sla": int(latency["latency_p95_ms"] < 200),
        }
        mlflow.log_metrics(metrics)

        logger.info(
            f"\nMetrics:\n"
            f"  F1 (weighted)  : {f1:.4f}\n"
            f"  F1 (drowsy)    : {f1_drowsy:.4f}\n"
            f"  Accuracy       : {accuracy:.4f}\n"
            f"  AUC-ROC        : {auc_roc:.4f}\n"
            f"  Latency P95    : {latency['latency_p95_ms']}ms\n"
        )

        # ── Log classification report ─────────────────────────────────────────
        report = classification_report(
            y_test, y_pred, target_names=["Alert", "Drowsy"]
        )
        logger.info(f"\nClassification Report:\n{report}")
        mlflow.log_text(report, "classification_report.txt")

        # ── Plots ─────────────────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(
            cm,
            "reports/confusion_matrix_xgb.png",
            f"XGBoost Confusion Matrix (F1={f1:.3f})"
        )
        mlflow.log_artifact(cm_path)

        fi_path = plot_feature_importance(model, "reports/feature_importance_xgb.png")
        mlflow.log_artifact(fi_path)

        # ── Log model with MLflow (native XGBoost flavour) ────────────────────
        # This enables mlflow.xgboost.load_model() later
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=pd.DataFrame([dict(zip(FEATURE_COLS, X_test[0]))]),
        )

        # ── Also save as pickle for DVC tracking ──────────────────────────────
        os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
        with open(MODEL_OUTPUT, "wb") as f:
            pickle.dump({"model": model, "feature_cols": FEATURE_COLS,
                         "mlflow_run_id": run_id}, f)
        logger.info(f"Model saved: {MODEL_OUTPUT}")

        # ── Save metrics for DVC ──────────────────────────────────────────────
        dvc_metrics = {
            "model": "xgboost",
            "mlflow_run_id": run_id,
            "git_commit": git_sha,
            **metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
        with open(METRICS_OUTPUT, "w") as f:
            json.dump(dvc_metrics, f, indent=2)
        logger.info(f"DVC metrics saved: {METRICS_OUTPUT}")

        mlflow.log_artifact(METRICS_OUTPUT)

        logger.info(
            f"\n{'='*60}\n"
            f"XGBOOST TRAINING COMPLETE\n"
            f"  MLflow run ID  : {run_id}\n"
            f"  F1 (weighted)  : {f1:.4f}\n"
            f"  AUC-ROC        : {auc_roc:.4f}\n"
            f"  Latency P95    : {latency['latency_p95_ms']}ms\n"
            f"  Model saved    : {MODEL_OUTPUT}\n"
            f"{'='*60}"
        )

    return dvc_metrics


if __name__ == "__main__":
    params = load_params()
    metrics = train(params)
    sys.exit(0)
