"""
airflow/plugins/data_quality_operator.py
─────────────────────────────────────────
Custom Airflow operator for reusable data quality checks.
Raises AirflowException if any check fails, which triggers
task failure and retry logic in the DAG.

Usage in a DAG:
    from plugins.data_quality_operator import DataQualityOperator

    check = DataQualityOperator(
        task_id="check_features",
        filepath="data/features/features.csv",
        checks=["exists", "not_empty", "has_both_labels", "no_nulls_in_features"],
    )
"""

import logging
import os

import pandas as pd
from airflow.exceptions import AirflowException
from airflow.models import BaseOperator

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]


class DataQualityOperator(BaseOperator):
    """
    Custom operator to run data quality checks on pipeline outputs.

    Supported checks:
      - 'exists'               : file/directory exists
      - 'not_empty'            : CSV has at least 1 row
      - 'has_both_labels'      : label column has both 0 and 1
      - 'no_nulls_in_features' : feature columns have no NaN values
      - 'schema'               : all required columns present
    """

    VALID_CHECKS = {
        "exists",
        "not_empty",
        "has_both_labels",
        "no_nulls_in_features",
        "schema",
    }

    def __init__(self, filepath: str, checks: list, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.checks = checks

        invalid = set(checks) - self.VALID_CHECKS
        if invalid:
            raise ValueError(f"Unknown checks: {invalid}. Valid: {self.VALID_CHECKS}")

    def execute(self, context):
        failures = []

        for check in self.checks:
            try:
                passed, message = self._run_check(check)
                if passed:
                    logger.info(f"✓ Check '{check}' passed: {message}")
                else:
                    logger.error(f"✗ Check '{check}' FAILED: {message}")
                    failures.append(f"{check}: {message}")
            except Exception as e:
                failures.append(f"{check}: exception — {str(e)}")

        if failures:
            raise AirflowException(
                f"DataQualityOperator failed {len(failures)} check(s):\n"
                + "\n".join(f"  • {f}" for f in failures)
            )

        logger.info(f"All {len(self.checks)} quality checks passed for {self.filepath}")
        return {"filepath": self.filepath, "checks_passed": self.checks}

    def _run_check(self, check: str) -> tuple:
        if check == "exists":
            exists = os.path.exists(self.filepath)
            return exists, f"{'found' if exists else 'NOT found'}: {self.filepath}"

        if check == "not_empty":
            if not os.path.isfile(self.filepath):
                return False, f"File not found: {self.filepath}"
            df = pd.read_csv(self.filepath, nrows=1)
            size = os.path.getsize(self.filepath)
            return size > 100, f"file size = {size} bytes"

        if check == "has_both_labels":
            if not os.path.isfile(self.filepath):
                return False, "File not found"
            df = pd.read_csv(self.filepath, usecols=["label"])
            n_unique = df["label"].nunique()
            return n_unique >= 2, f"unique labels = {n_unique} (need ≥ 2)"

        if check == "no_nulls_in_features":
            if not os.path.isfile(self.filepath):
                return False, "File not found"
            df = pd.read_csv(self.filepath)
            available = [c for c in FEATURE_COLS if c in df.columns]
            null_counts = df[available].isnull().sum()
            bad_cols = null_counts[null_counts > 0].to_dict()
            if bad_cols:
                return False, f"Null values found: {bad_cols}"
            return True, f"No nulls in {len(available)} feature columns"

        if check == "schema":
            if not os.path.isfile(self.filepath):
                return False, "File not found"
            df = pd.read_csv(self.filepath, nrows=0)  # just header
            missing = [c for c in (FEATURE_COLS + ["label"]) if c not in df.columns]
            if missing:
                return False, f"Missing columns: {missing}"
            return True, f"All {len(FEATURE_COLS)+1} required columns present"

        return False, f"Unknown check: {check}"
