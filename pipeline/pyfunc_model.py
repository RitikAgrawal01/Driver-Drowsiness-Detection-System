"""MLflow pyfunc wrapper: one interface for any champion — an ML model or the PERCLOS rule.
The fitted scaler/model, the feature order, the decision threshold and the window length
travel together inside the registered model."""

import mlflow.pyfunc
import numpy as np
import pandas as pd


class DrowsinessModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, model_type: str, features: list, threshold: float, window_s: int):
        self.estimator = estimator          # None for the rule
        self.model_type = model_type        # "rule" | "logreg" | "svm" | "xgboost"
        self.features = list(features)
        self.threshold = float(threshold)
        self.window_s = int(window_s)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        X = model_input[self.features]
        if self.model_type == "rule":
            s = X["perclos"].to_numpy()
        elif self.model_type == "svm":
            s = self.estimator.decision_function(X)
        else:
            s = self.estimator.predict_proba(X)[:, 1]
        s = np.asarray(s, dtype=float)
        return pd.DataFrame({"score": s, "drowsy": (s >= self.threshold).astype(int)})
