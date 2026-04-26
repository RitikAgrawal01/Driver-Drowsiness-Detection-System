"""
tests/unit/test_retrain_dag.py
────────────────────────────────
Unit tests for the retraining DAG task functions.
All external dependencies (MLflow, Prometheus, Airflow, model server)
are mocked so tests run without any services running.

Run:  pytest tests/unit/test_retrain_dag.py -v
"""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ── Make dag importable without Airflow installed ─────────────────────────────
# We test the pure Python logic inside each task function directly,
# not the Airflow operator wrappers.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../airflow/dags")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../airflow/scripts")
)


# ── Shared mock context ───────────────────────────────────────────────────────

def make_context(xcom_data=None, conf=None):
    """
    Build a minimal Airflow task context mock.
    xcom_data: dict of {task_id: {key: value}} for xcom_pull
    conf:      dict for dag_run.conf
    """
    xcom_data = xcom_data or {}
    ti = MagicMock()

    def xcom_pull(task_ids=None, key=None):
        return xcom_data.get(task_ids, {}).get(key)

    ti.xcom_pull.side_effect = xcom_pull
    ti.xcom_push = MagicMock()

    dag_run = MagicMock()
    dag_run.conf = conf or {}

    return {
        "ti": ti,
        "dag_run": dag_run,
        "run_id": "test_run_001",
        "execution_date": "2026-04-15T02:00:00+00:00",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: task_check_retrain_conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckRetrainConditions:

    def test_force_retrain_bypasses_checks(self):
        """force_retrain=True should return True without checking Prometheus or MLflow."""
        from dag_retrain import task_check_retrain_conditions
        ctx = make_context(conf={"force_retrain": True})

        with patch("dag_retrain.requests.get") as mock_get, \
             patch("dag_retrain.mlflow") as mock_mlflow:
            result = task_check_retrain_conditions(**ctx)

        assert result is True
        mock_get.assert_not_called()   # Prometheus NOT checked
        ctx["ti"].xcom_push.assert_any_call(key="retrain_reason", value="forced")

    def test_high_drift_triggers_retrain(self):
        """Drift score above threshold should return True."""
        from dag_retrain import task_check_retrain_conditions
        ctx = make_context()

        mock_prom_response = MagicMock()
        mock_prom_response.json.return_value = {
            "data": {"result": [{"value": ["1715000000", "0.22"]}]}
        }

        with patch("dag_retrain.requests.get", return_value=mock_prom_response), \
             patch("dag_retrain.mlflow") as mock_mlflow:
            mock_mlflow.MlflowClient.return_value.get_latest_versions.return_value = []
            result = task_check_retrain_conditions(**ctx)

        assert result is True

    def test_low_drift_no_retrain(self):
        """Drift score below threshold with healthy model should return False."""
        from dag_retrain import task_check_retrain_conditions
        ctx = make_context()

        # Low drift score
        mock_prom_response = MagicMock()
        mock_prom_response.json.return_value = {
            "data": {"result": [{"value": ["1715000000", "0.05"]}]}
        }

        # Healthy production model
        mock_version = MagicMock()
        mock_version.run_id = "abc123"
        mock_run = MagicMock()
        mock_run.data.metrics = {"f1_weighted": 0.92}

        with patch("dag_retrain.requests.get", return_value=mock_prom_response), \
             patch("dag_retrain.mlflow") as mock_mlflow:
            mock_mlflow.MlflowClient.return_value.get_latest_versions.return_value = [mock_version]
            mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run
            result = task_check_retrain_conditions(**ctx)

        assert result is False

    def test_no_production_model_triggers_retrain(self):
        """If no Production model exists, retraining should always run."""
        from dag_retrain import task_check_retrain_conditions
        ctx = make_context()

        mock_prom_response = MagicMock()
        mock_prom_response.json.return_value = {
            "data": {"result": [{"value": ["1715000000", "0.03"]}]}
        }

        with patch("dag_retrain.requests.get", return_value=mock_prom_response), \
             patch("dag_retrain.mlflow") as mock_mlflow:
            # No production model
            mock_mlflow.MlflowClient.return_value.get_latest_versions.return_value = []
            result = task_check_retrain_conditions(**ctx)

        assert result is True

    def test_prometheus_unreachable_continues(self):
        """If Prometheus is down, function should not raise — just continue."""
        from dag_retrain import task_check_retrain_conditions
        ctx = make_context()

        with patch("dag_retrain.requests.get", side_effect=Exception("Connection refused")), \
             patch("dag_retrain.mlflow") as mock_mlflow:
            mock_mlflow.MlflowClient.return_value.get_latest_versions.return_value = []
            # Should not raise
            result = task_check_retrain_conditions(**ctx)

        # With no production model found, should trigger
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: task_evaluate_models
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateModels:

    def _make_eval_context(self, xgb_f1=0.92, svm_f1=0.88, prod_f1=0.89):
        return make_context(xcom_data={
            "t3_train_xgboost": {
                "xgb_metrics": {"f1_weighted": xgb_f1, "auc_roc": 0.96},
                "xgb_run_id":  "run_xgb_001",
            },
            "t4_train_svm": {
                "svm_metrics": {"f1_weighted": svm_f1, "auc_roc": 0.93},
                "svm_run_id":  "run_svm_001",
            },
            "t0_check_retrain": {
                "prod_f1": prod_f1,
            },
        })

    def test_xgboost_wins_when_higher_f1(self):
        from dag_retrain import task_evaluate_models
        ctx = self._make_eval_context(xgb_f1=0.92, svm_f1=0.88)

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        # Check winner is XGBoost
        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        assert push_calls.get("winner") == "XGBoost"

    def test_svm_wins_when_higher_f1(self):
        from dag_retrain import task_evaluate_models
        ctx = self._make_eval_context(xgb_f1=0.85, svm_f1=0.91)

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        assert push_calls.get("winner") == "SVM"

    def test_should_promote_when_improvement_sufficient(self):
        """New model beats production by more than MIN_F1_TO_PROMOTE (0.02)."""
        from dag_retrain import task_evaluate_models
        # xgb=0.94, prod=0.89 → improvement=0.05 > 0.02 → should promote
        ctx = self._make_eval_context(xgb_f1=0.94, svm_f1=0.88, prod_f1=0.89)

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        assert push_calls.get("should_promote") is True

    def test_should_not_promote_when_improvement_too_small(self):
        """New model barely beats production — below MIN_F1_TO_PROMOTE."""
        from dag_retrain import task_evaluate_models
        # xgb=0.900, prod=0.895 → improvement=0.005 < 0.02 → don't promote
        ctx = self._make_eval_context(xgb_f1=0.900, svm_f1=0.88, prod_f1=0.895)

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        assert push_calls.get("should_promote") is False

    def test_should_promote_when_no_production_model(self):
        """If no production model exists, always promote."""
        from dag_retrain import task_evaluate_models
        ctx = make_context(xcom_data={
            "t3_train_xgboost": {
                "xgb_metrics": {"f1_weighted": 0.85, "auc_roc": 0.90},
                "xgb_run_id":  "run_xgb_002",
            },
            "t4_train_svm": {
                "svm_metrics": {"f1_weighted": 0.80, "auc_roc": 0.88},
                "svm_run_id":  "run_svm_002",
            },
            "t0_check_retrain": {"prod_f1": None},  # No production model
        })

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        assert push_calls.get("should_promote") is True

    def test_comparison_dict_has_required_keys(self):
        """The comparison XCom dict must contain all required keys."""
        from dag_retrain import task_evaluate_models
        ctx = self._make_eval_context()

        with patch("dag_retrain.mlflow"):
            task_evaluate_models(**ctx)

        push_calls = {
            call[1]["key"]: call[1]["value"]
            for call in ctx["ti"].xcom_push.call_args_list
        }
        comparison = push_calls.get("comparison", {})
        required_keys = {
            "xgb_f1", "svm_f1", "winner", "winner_f1",
            "prod_f1", "improvement", "should_promote",
        }
        assert required_keys.issubset(comparison.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Tests: task_promote_if_better
# ─────────────────────────────────────────────────────────────────────────────

class TestPromoteIfBetter:

    def test_returns_false_when_should_not_promote(self):
        from dag_retrain import task_promote_if_better
        ctx = make_context(xcom_data={
            "t5_evaluate_models": {
                "comparison": {
                    "should_promote": False,
                    "winner": "XGBoost",
                    "winner_f1": 0.88,
                    "winner_run_id": "run_001",
                    "improvement": 0.005,
                }
            }
        })

        result = task_promote_if_better(**ctx)
        assert result is False

    def test_returns_false_when_no_run_id(self):
        from dag_retrain import task_promote_if_better
        ctx = make_context(xcom_data={
            "t5_evaluate_models": {
                "comparison": {
                    "should_promote": True,
                    "winner": "XGBoost",
                    "winner_f1": 0.93,
                    "winner_run_id": None,   # missing
                    "improvement": 0.04,
                }
            }
        })

        result = task_promote_if_better(**ctx)
        assert result is False

    def test_promotes_successfully(self):
        from dag_retrain import task_promote_if_better
        ctx = make_context(xcom_data={
            "t5_evaluate_models": {
                "comparison": {
                    "should_promote": True,
                    "winner": "XGBoost",
                    "winner_f1": 0.94,
                    "winner_run_id": "run_abc",
                    "improvement": 0.05,
                }
            },
            "t0_check_retrain": {"retrain_reason": "drift"},
        })

        mock_version = MagicMock()
        mock_version.run_id = "run_abc"
        mock_version.version = "5"
        mock_version.current_stage = "Staging"

        with patch("dag_retrain.mlflow") as mock_mlflow:
            client = mock_mlflow.MlflowClient.return_value
            client.search_model_versions.return_value = [mock_version]
            client.transition_model_version_stage.return_value = None

            result = task_promote_if_better(**ctx)

        assert result is True
        # Verify promotion was called with Production stage
        client.transition_model_version_stage.assert_called_with(
            name="drowsiness_classifier",
            version="5",
            stage="Production",
        )

    def test_mlflow_unreachable_raises(self):
        """If MLflow is unreachable during promotion, task should raise."""
        from dag_retrain import task_promote_if_better
        ctx = make_context(xcom_data={
            "t5_evaluate_models": {
                "comparison": {
                    "should_promote": True,
                    "winner": "XGBoost",
                    "winner_f1": 0.94,
                    "winner_run_id": "run_abc",
                    "improvement": 0.05,
                }
            },
            "t0_check_retrain": {"retrain_reason": "drift"},
        })

        with patch("dag_retrain.mlflow") as mock_mlflow:
            mock_mlflow.MlflowClient.side_effect = Exception("MLflow unreachable")
            with pytest.raises(Exception, match="MLflow unreachable"):
                task_promote_if_better(**ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: task_pipeline_report
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineReport:

    def test_report_saved_to_disk(self, tmp_path):
        from dag_retrain import task_pipeline_report
        import dag_retrain
        # Override REPORTS_DIR to temp path
        original = dag_retrain.REPORTS_DIR
        dag_retrain.REPORTS_DIR = str(tmp_path)

        ctx = make_context(xcom_data={
            "t0_check_retrain":    {"retrain_reason": "drift_score=0.18"},
            "t1_pull_data":        {"n_samples": 4000},
            "t3_train_xgboost":    {"xgb_metrics": {"f1_weighted": 0.91}},
            "t4_train_svm":        {"svm_metrics": {"f1_weighted": 0.87}},
            "t5_evaluate_models":  {"comparison": {"winner": "XGBoost", "winner_f1": 0.91}},
            "t6_promote_if_better":{"promoted": True, "promoted_version": "6"},
        })

        with patch("dag_retrain.subprocess.run"):
            task_pipeline_report(**ctx)

        dag_retrain.REPORTS_DIR = original

        # Verify JSON file was written
        report_files = list(tmp_path.glob("retrain_report_*.json"))
        assert len(report_files) == 1

        with open(report_files[0]) as f:
            report = json.load(f)

        assert report["retrain_run"]["retrain_reason"] == "drift_score=0.18"
        assert report["promotion"]["promoted"] is True
        assert report["promotion"]["promoted_version"] == "6"

    def test_report_has_required_sections(self, tmp_path):
        from dag_retrain import task_pipeline_report
        import dag_retrain
        dag_retrain.REPORTS_DIR = str(tmp_path)

        ctx = make_context(xcom_data={
            "t0_check_retrain":    {"retrain_reason": "forced"},
            "t1_pull_data":        {"n_samples": 2000},
            "t3_train_xgboost":    {"xgb_metrics": {}},
            "t4_train_svm":        {"svm_metrics": {}},
            "t5_evaluate_models":  {"comparison": {}},
            "t6_promote_if_better":{"promoted": False, "promoted_version": None},
        })

        with patch("dag_retrain.subprocess.run"):
            task_pipeline_report(**ctx)

        report_files = list(tmp_path.glob("retrain_report_*.json"))
        with open(report_files[0]) as f:
            report = json.load(f)

        required_sections = {"retrain_run", "new_models", "comparison",
                             "promotion", "timestamp"}
        assert required_sections.issubset(report.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Tests: prometheus_webhook trigger logic
# ─────────────────────────────────────────────────────────────────────────────

class TestPrometheusWebhook:

    def test_trigger_airflow_dag_success(self):
        from tools.prometheus_webhook import trigger_airflow_dag
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"dag_run_id": "test_run", "state": "queued"}

        with patch("tools.prometheus_webhook.requests.post", return_value=mock_resp):
            result = trigger_airflow_dag("test_reason")

        assert result is True

    def test_trigger_airflow_dag_connection_error(self):
        from tools.prometheus_webhook import trigger_airflow_dag

        with patch("tools.prometheus_webhook.requests.post",
                   side_effect=Exception("Connection refused")):
            result = trigger_airflow_dag("test_reason")

        assert result is False

    def test_trigger_airflow_dag_409_conflict(self):
        """409 means DAG already running — should return False gracefully."""
        from tools.prometheus_webhook import trigger_airflow_dag
        mock_resp = MagicMock()
        mock_resp.status_code = 409

        with patch("tools.prometheus_webhook.requests.post", return_value=mock_resp):
            result = trigger_airflow_dag("test_reason")

        assert result is False
