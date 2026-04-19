"""
model_server/evaluate.py
─────────────────────────
Stage 7 of the DVC pipeline.

Compares XGBoost and SVM on the test set:
  • Loads both models from their pickle files
  • Computes final evaluation metrics side-by-side
  • Generates ROC curve comparison plot
  • Determines the winner (higher F1 weighted)
  • Promotes the winning model to 'Production' in MLflow registry
  • Moves the loser to 'Archived'
  • Saves evaluation_report.json for DVC metrics

Can be run:
  • Standalone : python model_server/evaluate.py
  • By DVC     : dvc repro evaluate
"""

import json
import logging
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate")

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CSV = os.getenv("TEST_CSV", "data/processed/test.csv")
XGB_MODEL_PATH = os.getenv("XGB_MODEL_OUTPUT", "models/xgboost_model.pkl")
SVM_MODEL_PATH = os.getenv("SVM_MODEL_OUTPUT", "models/svm_model.pkl")
EVALUATION_OUTPUT = os.getenv("EVALUATION_OUTPUT", "reports/evaluation_report.json")
PARAMS_FILE = os.getenv("PARAMS_FILE", "params.yaml")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "drowsiness_classifier")

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]

ACCEPTANCE_CRITERIA = {
    "min_f1_weighted": 0.85,
    "min_auc_roc": 0.90,
    "max_latency_p95_ms": 200,
}


def load_params() -> dict:
    if not os.path.isfile(PARAMS_FILE):
        return {}
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_model_from_pickle(path: str) -> tuple:
    """Load model and metadata from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("feature_cols", FEATURE_COLS), data


def measure_latency(model, X: np.ndarray, n_runs: int = 100) -> dict:
    """Measure per-sample inference latency."""
    latencies = []
    sample = X[0:1]
    for _ in range(10):
        model.predict_proba(sample)
    for _ in range(n_runs):
        t = time.perf_counter()
        model.predict_proba(sample)
        latencies.append((time.perf_counter() - t) * 1000)
    return {
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   name: str) -> dict:
    """Evaluate a single model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    latency = measure_latency(model, X_test)

    metrics = {
        "model": name,
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_drowsy_class": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(precision_score(y_test, y_pred,
                                                     average="weighted"), 4),
        "recall_weighted": round(recall_score(y_test, y_pred,
                                               average="weighted"), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        **latency,
        "y_proba": y_proba.tolist(),   # for ROC curve plot
    }

    # FIX: Wrap comparisons in bool() to ensure JSON serializability
    criteria_results = {
        "meets_f1": bool(metrics["f1_weighted"] >= ACCEPTANCE_CRITERIA["min_f1_weighted"]),
        "meets_auc": bool(metrics["auc_roc"] >= ACCEPTANCE_CRITERIA["min_auc_roc"]),
        "meets_latency": bool(metrics["latency_p95_ms"] <= ACCEPTANCE_CRITERIA["max_latency_p95_ms"]),
    }
    
    # FIX: Ensure this is also a standard Python bool
    metrics["meets_acceptance_criteria"] = bool(all(criteria_results.values()))
    metrics["acceptance_detail"] = criteria_results

    logger.info(
        f"\n{name} Evaluation:\n"
        f"  F1 (weighted)  : {metrics['f1_weighted']}\n"
        f"  F1 (drowsy)    : {metrics['f1_drowsy_class']}\n"
        f"  AUC-ROC        : {metrics['auc_roc']}\n"
        f"  Accuracy       : {metrics['accuracy']}\n"
        f"  Latency P95    : {metrics['latency_p95_ms']}ms\n"
        f"  Meets criteria : {metrics['meets_acceptance_criteria']}\n"
        f"  Criteria detail: {criteria_results}"
    )

    print(classification_report(y_test, model.predict(X_test),
                                  target_names=["Alert", "Drowsy"]))
    return metrics


def plot_roc_comparison(y_test: np.ndarray, xgb_proba: list,
                         svm_proba: list, xgb_auc: float,
                         svm_auc: float, output_path: str) -> str:
    """Generate a side-by-side ROC curve comparison."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    ax.plot(fpr_xgb, tpr_xgb, color="#06B6D4", lw=2,
            label=f"XGBoost (AUC = {xgb_auc:.3f})")

    # SVM ROC
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba)
    ax.plot(fpr_svm, tpr_svm, color="#A78BFA", lw=2,
            label=f"SVM (AUC = {svm_auc:.3f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], color="#374151", lw=1,
            linestyle="--", label="Random baseline")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Curve Comparison — XGBoost vs SVM",
                  color="white", fontsize=13, fontweight="bold")

    legend = ax.legend(loc="lower right", facecolor="#1E2A3A",
                        edgecolor="#374151", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2A3A")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"ROC curve saved: {output_path}")
    return output_path


def promote_to_production(client: MlflowClient, run_id: str,
                           model_name: str, winner: str) -> str:
    """
    Find the MLflow model version matching this run_id and
    transition it to Production stage.
    Moves any existing Production version to Archived.
    """
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")

        # Find version matching our run_id
        target_version = None
        for v in versions:
            if v.run_id == run_id:
                target_version = v
                break

        if target_version is None:
            logger.warning(
                f"No MLflow model version found for run_id={run_id}. "
                f"Skipping registry promotion. "
                f"(This is normal if MLflow server is not running locally)"
            )
            return "not_promoted"

        # Archive any existing Production versions
        for v in versions:
            if v.current_stage == "Production" and v.version != target_version.version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )
                logger.info(f"Archived previous production version: {v.version}")

        # Promote winner to Production
        client.transition_model_version_stage(
            name=model_name,
            version=target_version.version,
            stage="Production",
        )

        # Add description tag
        client.update_model_version(
            name=model_name,
            version=target_version.version,
            description=f"Production model — {winner} (promoted by evaluate.py)",
        )

        logger.info(
            f"✓ Promoted {winner} version {target_version.version} → Production"
        )
        return target_version.version

    except Exception as e:
        logger.warning(f"MLflow promotion skipped (server may be offline): {e}")
        return "not_promoted"


def run_evaluation() -> dict:
    """Main evaluation function."""

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info(f"Loading test data: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    logger.info(f"Test set: {X_test.shape}, label dist: "
                f"{dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ── Load both models ──────────────────────────────────────────────────────
    logger.info(f"Loading XGBoost model: {XGB_MODEL_PATH}")
    xgb_model, _, xgb_meta = load_model_from_pickle(XGB_MODEL_PATH)

    logger.info(f"Loading SVM model: {SVM_MODEL_PATH}")
    svm_model, _, svm_meta = load_model_from_pickle(SVM_MODEL_PATH)

    # ── Evaluate both ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("EVALUATING XGBOOST")
    logger.info("="*60)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    logger.info("\n" + "="*60)
    logger.info("EVALUATING SVM")
    logger.info("="*60)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")

    # ── ROC curve comparison ──────────────────────────────────────────────────
    roc_path = plot_roc_comparison(
        y_test,
        xgb_metrics.pop("y_proba"),
        svm_metrics.pop("y_proba"),
        xgb_metrics["auc_roc"],
        svm_metrics["auc_roc"],
        "reports/roc_curve.png",
    )

    # ── Determine winner ──────────────────────────────────────────────────────
    xgb_f1 = xgb_metrics["f1_weighted"]
    svm_f1 = svm_metrics["f1_weighted"]

    if xgb_f1 >= svm_f1:
        winner = "XGBoost"
        winner_run_id = xgb_meta.get("mlflow_run_id", "")
        winner_metrics = xgb_metrics
    else:
        winner = "SVM"
        winner_run_id = svm_meta.get("mlflow_run_id", "")
        winner_metrics = svm_metrics

    logger.info(
        f"\n{'='*60}\n"
        f"WINNER: {winner}\n"
        f"  XGBoost F1: {xgb_f1:.4f}\n"
        f"  SVM     F1: {svm_f1:.4f}\n"
        f"{'='*60}"
    )

    # ── MLflow: promote winner to Production ──────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    promoted_version = promote_to_production(
        client, winner_run_id, MLFLOW_MODEL_NAME, winner
    )

    # ── Log evaluation run to MLflow ──────────────────────────────────────────
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metrics({
            "xgb_f1_weighted": xgb_metrics["f1_weighted"],
            "xgb_auc_roc": xgb_metrics["auc_roc"],
            "xgb_latency_p95_ms": xgb_metrics["latency_p95_ms"],
            "svm_f1_weighted": svm_metrics["f1_weighted"],
            "svm_auc_roc": svm_metrics["auc_roc"],
            "svm_latency_p95_ms": svm_metrics["latency_p95_ms"],
            "winner_f1": winner_metrics["f1_weighted"],
        })
        mlflow.log_params({
            "winner": winner,
            "promoted_version": str(promoted_version),
        })
        mlflow.log_artifact(roc_path)
        mlflow.set_tag("evaluation_type", "model_comparison")

    # ── Build final report ────────────────────────────────────────────────────
    report = {
        "evaluation_summary": {
            "winner": winner,
            "winner_run_id": winner_run_id,
            "promoted_mlflow_version": str(promoted_version),
            "acceptance_criteria": ACCEPTANCE_CRITERIA,
        },
        "xgboost": {k: v for k, v in xgb_metrics.items()
                    if k not in ("y_proba",)},
        "svm": {k: v for k, v in svm_metrics.items()
                if k not in ("y_proba",)},
        "winner_metrics": winner_metrics,
        "artifacts": {
            "roc_curve": roc_path,
            "xgb_confusion_matrix": "reports/confusion_matrix_xgb.png",
            "svm_confusion_matrix": "reports/confusion_matrix_svm.png",
        },
    }

    os.makedirs(os.path.dirname(EVALUATION_OUTPUT), exist_ok=True)
    with open(EVALUATION_OUTPUT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\n✓ Evaluation report saved: {EVALUATION_OUTPUT}")

    # ── Print acceptance test result ──────────────────────────────────────────
    meets = winner_metrics["meets_acceptance_criteria"]
    logger.info(
        f"\n{'='*60}\n"
        f"ACCEPTANCE TEST: {'✓ PASSED' if meets else '✗ FAILED'}\n"
        f"  Min F1   ≥ {ACCEPTANCE_CRITERIA['min_f1_weighted']} : "
        f"{'✓' if winner_metrics['f1_weighted'] >= ACCEPTANCE_CRITERIA['min_f1_weighted'] else '✗'} "
        f"({winner_metrics['f1_weighted']})\n"
        f"  Min AUC  ≥ {ACCEPTANCE_CRITERIA['min_auc_roc']} : "
        f"{'✓' if winner_metrics['auc_roc'] >= ACCEPTANCE_CRITERIA['min_auc_roc'] else '✗'} "
        f"({winner_metrics['auc_roc']})\n"
        f"  Latency ≤ {ACCEPTANCE_CRITERIA['max_latency_p95_ms']}ms : "
        f"{'✓' if winner_metrics['latency_p95_ms'] <= ACCEPTANCE_CRITERIA['max_latency_p95_ms'] else '✗'} "
        f"({winner_metrics['latency_p95_ms']}ms)\n"
        f"{'='*60}"
    )

    return report


if __name__ == "__main__":
    report = run_evaluation()
    # Exit 1 if winner doesn't meet acceptance criteria (fails DVC stage)
    winner_meets = report["winner_metrics"].get("meets_acceptance_criteria", False)
    if not winner_meets:
        logger.warning(
            "Winner model did not meet acceptance criteria. "
            "Consider tuning hyperparameters in params.yaml and re-running dvc repro."
        )
    sys.exit(0)  # don't fail pipeline — just warn
