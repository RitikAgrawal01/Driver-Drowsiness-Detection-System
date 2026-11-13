"""
model_server/model_loader.py
Loads the registered champion (`models:/<name>@<alias>`, both from params.yaml) as an MLflow
pyfunc model. The pyfunc wrapper carries its own scaler/model, feature order, decision
threshold and window length, so any model family (XGBoost, SVM, logistic regression) serves
the same way.

No pickle fallback and no silent defaults: if the model cannot be loaded the server reports
not-ready (503), and an inference error is an error — never an "alert" prediction.
"""

import logging
import threading
import time

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from ddd_core.config import load_params

logger = logging.getLogger("model_loader")


class ModelNotLoaded(RuntimeError):
    pass


class ModelStore:
    """Holds the current model; reloads are atomic (old model keeps serving until the new one is ready)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._meta: dict = {}

    def load(self) -> dict:
        ml = load_params()["mlflow"]
        name, alias = ml["model_name"], ml["alias"]
        uri = f"models:/{name}@{alias}"
        logger.info("Loading %s from %s", uri, mlflow.get_tracking_uri())
        model = mlflow.pyfunc.load_model(uri)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        run = client.get_run(mv.run_id)
        md = model.metadata.metadata or {}
        meta = {
            "model_name": name,
            "alias": alias,
            "model_version": str(mv.version),
            "mlflow_run_id": mv.run_id,
            "model_type": md.get("model_type"),
            "features": md.get("features"),
            "threshold": md.get("threshold"),
            "window_s": md.get("window_s"),
            "target_fps": md.get("target_fps"),
            "loso_metrics": {k: v for k, v in run.data.metrics.items() if k.startswith("loso_")},
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        missing = [k for k in ("features", "threshold", "window_s") if meta[k] is None]
        if missing:
            raise RuntimeError(f"Registered model lacks metadata {missing}; refusing to serve it")
        with self._lock:
            self._model, self._meta = model, meta
        logger.info(
            "Serving %s v%s (%s, window %s s, threshold %.4f)",
            name,
            meta["model_version"],
            meta["model_type"],
            meta["window_s"],
            meta["threshold"],
        )
        return meta

    def is_loaded(self) -> bool:
        return self._model is not None

    def meta(self) -> dict:
        return dict(self._meta)

    def predict(self, features: dict) -> dict:
        with self._lock:
            model, meta = self._model, self._meta
        if model is None:
            raise ModelNotLoaded("model not loaded")
        t0 = time.perf_counter()
        out = model.predict(pd.DataFrame([features])[meta["features"]])
        latency_ms = (time.perf_counter() - t0) * 1000
        score, drowsy = float(out["score"].iloc[0]), int(out["drowsy"].iloc[0])
        return {
            "state": "drowsy" if drowsy else "alert",
            "score": round(score, 6),
            "threshold": meta["threshold"],
            "model_version": meta["model_version"],
            "model_type": meta["model_type"],
            "window_s": meta["window_s"],
            "inference_latency_ms": round(latency_ms, 3),
        }


store = ModelStore()
