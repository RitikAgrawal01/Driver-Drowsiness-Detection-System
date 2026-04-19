"""
model_server/train_svm.py
──────────────────────────
Stage 6 of the DVC pipeline.

Trains an SVM classifier (RBF kernel) on the drowsiness feature dataset.
Logs everything to MLflow for comparison with XGBoost.

SVM requires feature scaling (StandardScaler) — the scaler is saved
alongside the model so inference uses the same scale.

Can be run:
  • Standalone : python model_server/train_svm.py
  • By DVC     : dvc repro train_svm
"""

import json
import logging
import os
import pickle
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_svm")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CSV = os.getenv("TRAIN_CSV", "data/processed/train.csv")
TEST_CSV = os.getenv("TEST_CSV", "data/processed/test.csv")
MODEL_OUTPUT = os.getenv("SVM_MODEL_OUTPUT", "models/svm_model.pkl")
METRICS_OUTPUT = os.getenv("SVM_METRICS_OUTPUT", "reports/svm_metrics.json")
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
    if not os.path.isfile(PARAMS_FILE):
        return {}
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str) -> str:
    """Plot and save a styled confusion matrix (same style as XGBoost)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
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
    ax.set_title(title, color="#A78BFA", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2A3A")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def measure_inference_latency(pipeline, X_test: np.ndarray,
                               n_runs: int = 100) -> dict:
    """Measure single-sample inference latency."""
    latencies = []
    single = X_test[0:1]

    for _ in range(10):  # warm up
        pipeline.predict_proba(single)

    for _ in range(n_runs):
        t = time.perf_counter()
        pipeline.predict_proba(single)
        latencies.append((time.perf_counter() - t) * 1000)

    return {
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
    }


def train(params: dict) -> dict:
    """Main SVM training function."""
    svm_params = params.get("svm", {})
    feature_params = params.get("features", {})

    # ── Configure MLflow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values

    n_alert = int((y_train == 0).sum())
    n_drowsy = int((y_train == 1).sum())

    logger.info(
        f"Train: {X_train.shape} | Test: {X_test.shape}\n"
        f"  Train label dist: alert={n_alert}, drowsy={n_drowsy}"
    )

    with mlflow.start_run(run_name="svm_training") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Log parameters ────────────────────────────────────────────────────
        mlflow.log_params({
            "model_type": "SVM",
            "kernel": svm_params.get("kernel", "rbf"),
            "C": svm_params.get("C", 1.0),
            "gamma": svm_params.get("gamma", "scale"),
            "class_weight": svm_params.get("class_weight", "balanced"),
            "scaler": "StandardScaler",
            "probability": True,
            # Feature params
            "sliding_window_size": feature_params.get("sliding_window_size", 30),
            "ear_threshold": feature_params.get("ear_threshold", 0.25),
            # Dataset info
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(FEATURE_COLS),
            "feature_names": ",".join(FEATURE_COLS),
        })

        git_sha = get_git_commit()
        mlflow.set_tags({
            "git_commit": git_sha,
            "train_samples_alert": n_alert,
            "train_samples_drowsy": n_drowsy,
            "mlflow.note.content": f"SVM drowsiness classifier | Git: {git_sha}",
        })

        # ── Build sklearn Pipeline (scaler + SVM) ────────────────────────────
        # Pipeline ensures scaler is applied consistently during inference
        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=svm_params.get("kernel", "rbf"),
                C=svm_params.get("C", 1.0),
                gamma=svm_params.get("gamma", "scale"),
                class_weight=svm_params.get("class_weight", "balanced"),
                probability=True,   # needed for predict_proba + confidence score
                random_state=42,
            )),
        ])

        logger.info("Training SVM pipeline (StandardScaler + SVC)...")
        train_start = time.time()
        svm_pipeline.fit(X_train, y_train)
        train_time = time.time() - train_start
        logger.info(f"Training complete in {train_time:.1f}s")
        mlflow.log_metric("training_time_sec", round(train_time, 2))

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = svm_pipeline.predict(X_test)
        y_proba = svm_pipeline.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred, average="weighted")
        f1_drowsy = f1_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        auc_roc = roc_auc_score(y_test, y_proba)

        latency = measure_inference_latency(svm_pipeline, X_test)

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

        # ── Classification report ─────────────────────────────────────────────
        report = classification_report(y_test, y_pred,
                                        target_names=["Alert", "Drowsy"])
        logger.info(f"\nClassification Report:\n{report}")
        mlflow.log_text(report, "classification_report.txt")

        # ── Confusion matrix plot ─────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(
            cm,
            "reports/confusion_matrix_svm.png",
            f"SVM Confusion Matrix (F1={f1:.3f})"
        )
        mlflow.log_artifact(cm_path)

        # ── Log sklearn pipeline with MLflow ──────────────────────────────────
        # Log as sklearn model — includes scaler in the pipeline
        mlflow.sklearn.log_model(
            svm_pipeline,
            artifact_path="svm_model",
            registered_model_name=f"{MLFLOW_MODEL_NAME}_svm",
            input_example=pd.DataFrame(
                [dict(zip(FEATURE_COLS, X_test[0]))],
                columns=FEATURE_COLS
            ),
        )

        # ── Save as pickle for DVC ────────────────────────────────────────────
        os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
        with open(MODEL_OUTPUT, "wb") as f:
            pickle.dump({
                "model": svm_pipeline,
                "feature_cols": FEATURE_COLS,
                "mlflow_run_id": run_id,
                "scaler_included": True,  # scaler is inside the Pipeline
            }, f)
        logger.info(f"Model saved: {MODEL_OUTPUT}")

        # ── Save DVC metrics ──────────────────────────────────────────────────
        dvc_metrics = {
            "model": "svm",
            "mlflow_run_id": run_id,
            "git_commit": git_sha,
            **metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
        with open(METRICS_OUTPUT, "w") as f:
            json.dump(dvc_metrics, f, indent=2)

        mlflow.log_artifact(METRICS_OUTPUT)

        logger.info(
            f"\n{'='*60}\n"
            f"SVM TRAINING COMPLETE\n"
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
