"""
model_server/model_loader.py
──────────────────────────────
Loads the Production model from the MLflow Model Registry on startup.
Falls back to loading from local pickle file if MLflow is unavailable.

Used by model_server/main.py in the FastAPI lifespan event.
"""

import logging
import os
import pickle
from typing import Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
import numpy as np
import xgboost as xgb

logger = logging.getLogger("model_loader")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "drowsiness_classifier")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Local fallback paths (used if MLflow is unreachable)
XGB_PICKLE_PATH = os.getenv("XGB_MODEL_OUTPUT", "models/xgboost_model.pkl")
SVM_PICKLE_PATH = os.getenv("SVM_MODEL_OUTPUT", "models/svm_model.pkl")

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]

# Global model state — populated at startup
_model = None
_model_meta = {
    "model_name": MLFLOW_MODEL_NAME,
    "model_version": "unknown",
    "model_stage": MLFLOW_MODEL_STAGE,
    "algorithm": "unknown",
    "mlflow_run_id": "unknown",
    "source": "not_loaded",
    "feature_names": FEATURE_COLS,
}


def load_from_mlflow() -> bool:
    global _model, _model_meta
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        logger.info(f"Loading model from MLflow: {model_uri}")

        try:
            # Force MLflow to load it as a native XGBoost model
            _model = mlflow.xgboost.load_model(model_uri)
            logger.info("Successfully loaded native XGBoost booster.")
        except Exception as e:
            logger.warning(f"XGBoost native load failed: {e}. Trying generic pyfunc...")
            _model = mlflow.pyfunc.load_model(model_uri)

        # Get run info for metadata
        from mlflow import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions(
            MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE]
        )
        if versions:
            v = versions[0]
            run = client.get_run(v.run_id)
            algorithm = run.data.params.get("model_type", "unknown")
            _model_meta.update({
                "model_version": v.version,
                "model_stage": v.current_stage,
                "algorithm": algorithm,
                "mlflow_run_id": v.run_id,
                "source": "mlflow_registry",
                "metrics": {
                    k: v for k, v in run.data.metrics.items()
                    if k in ("f1_weighted", "auc_roc",
                             "latency_p95_ms", "accuracy")
                },
            })

        logger.info(
            f"✓ Model loaded from MLflow registry:\n"
            f"  Name     : {MLFLOW_MODEL_NAME}\n"
            f"  Stage    : {MLFLOW_MODEL_STAGE}\n"
            f"  Version  : {_model_meta['model_version']}\n"
            f"  Algorithm: {_model_meta['algorithm']}\n"
            f"  Run ID   : {_model_meta['mlflow_run_id']}"
        )
        return True

    except Exception as e:
        logger.warning(f"MLflow load failed: {e}")
        return False


def load_from_pickle() -> bool:
    """
    Fallback: load model from local pickle file.
    Tries XGBoost first, then SVM.
    """
    global _model, _model_meta

    for path, name in [(XGB_PICKLE_PATH, "XGBoost"), (SVM_PICKLE_PATH, "SVM")]:
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                _model = data["model"]
                _model_meta.update({
                    "algorithm": name,
                    "mlflow_run_id": data.get("mlflow_run_id", "unknown"),
                    "model_version": "local",
                    "source": f"pickle:{path}",
                })
                logger.info(
                    f"✓ Model loaded from pickle: {path} ({name})"
                )
                return True
            except Exception as e:
                logger.error(f"Pickle load failed for {path}: {e}")

    logger.error("No model could be loaded from MLflow or pickle!")
    return False


def load_model() -> bool:
    """
    Load model: try MLflow first, fall back to pickle.
    Called once at application startup.
    """
    if load_from_mlflow():
        return True
    logger.warning("Falling back to local pickle model...")
    return load_from_pickle()


def get_model():
    """Return the loaded model object."""
    return _model


def get_model_meta() -> dict:
    """Return model metadata dict."""
    return _model_meta.copy()


def is_loaded() -> bool:
    """Check if a model is currently loaded."""
    return _model is not None


import pandas as pd # Ensure this is imported at the top!

def predict(features: list) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded.")

    import time
    t_start = time.perf_counter()

    # Create a DataFrame with the exact feature names the model expects
    df = pd.DataFrame([features], columns=FEATURE_COLS)

    try:
        raw_output = _model.predict(df)
        
        if isinstance(_model, xgb.sklearn.XGBClassifier):
            proba = _model.predict_proba(np.array([features]))[0]
            confidence = float(proba[1])
        elif isinstance(_model, xgb.Booster):
            dmat = xgb.DMatrix(np.array([features]), feature_names=FEATURE_COLS)
            confidence = float(_model.predict(dmat)[0]) # Boosters return prob by default
        else:
            # Fallback for generic pyfunc
            res = _model.predict(pd.DataFrame([features], columns=FEATURE_COLS))
            confidence = float(res[0])

    except Exception as e:
        logger.error(f"Inference math failed: {e}")
        confidence = 0.0

    latency_ms = (time.perf_counter() - t_start) * 1000
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    state = "drowsy" if confidence >= threshold else "alert"

    return {
        "state": state,
        "confidence": round(confidence, 4),
        "inference_latency_ms": round(latency_ms, 3),
        "model_version": _model_meta.get("model_version", "unknown"),
    }