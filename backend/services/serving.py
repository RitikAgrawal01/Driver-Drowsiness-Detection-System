"""
What the backend is currently serving: the champion's metadata (window length, threshold,
version) and its drift baseline. Refreshed from the model server, so a model hot-reload is
picked up without restarting the backend.
"""

import json
import tempfile

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import DRIFT_BASELINE_LOADED, SERVED_MODEL_VERSION
from backend.services import model_client
from backend.services.drift_monitor import drift_monitor
from ddd_core.config import load_params

logger = get_logger("serving")


class ServingContext:
    def __init__(self):
        self.model: dict | None = None
        self.config_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.config_error is None

    async def refresh(self) -> None:
        info = await model_client.model_info()
        if info is None:
            return  # keep the last known model info; sessions will see model errors
        changed = self.model is None or info.get("model_version") != self.model.get("model_version")
        self.model = info
        want_fps = load_params()["serving"]["target_fps"]
        if info.get("target_fps") not in (None, want_fps):
            self.config_error = f"model trained at {info['target_fps']} fps but serving.target_fps={want_fps}"
            logger.error("config mismatch", detail=self.config_error)
        else:
            self.config_error = None
        if changed:
            SERVED_MODEL_VERSION.set(float(info["model_version"]))
            logger.info(
                "serving model",
                version=info["model_version"],
                window_s=info["window_s"],
                threshold=info["threshold"],
                model_type=info.get("model_type"),
            )
            self._load_baseline(info.get("mlflow_run_id"))

    def _load_baseline(self, run_id: str | None) -> None:
        if not run_id:
            DRIFT_BASELINE_LOADED.set(0)
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
            with tempfile.TemporaryDirectory() as d:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="drift_baseline.json", dst_path=d
                )
                with open(path) as f:
                    drift_monitor.set_baseline(json.load(f))
            DRIFT_BASELINE_LOADED.set(1)
        except Exception as e:
            DRIFT_BASELINE_LOADED.set(0)
            drift_monitor.set_baseline(None)
            logger.error("drift baseline not loaded; drift monitoring disabled", error=str(e))


serving = ServingContext()
