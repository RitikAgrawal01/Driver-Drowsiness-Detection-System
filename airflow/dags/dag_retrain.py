"""
airflow/dags/dag_retrain.py  — Phase 9 FIXED VERSION
─────────────────────────────────────────────────────
Key fixes applied:
  1. os.chdir("/app") in every task that touches the filesystem
  2. All file paths are ABSOLUTE (/app/...) — no relative paths
  3. MLFLOW env vars set at module level so they apply before any import
  4. GIT_PYTHON_REFRESH=quiet silences the git warning
  5. /tmp/mlflow created before any MLflow call
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

# ── Module-level env vars — applied before any MLflow import ──────────────────
os.environ["MLFLOW_TRACKING_URI"]        = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["GIT_PYTHON_REFRESH"]         = "quiet"   # silence git warning
os.makedirs("/tmp/mlflow", exist_ok=True)             # pre-create writable dir

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("dag_retrain")

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "drowsiness_classifier")
BACKEND_URL         = os.getenv("BACKEND_URL",        "http://backend:8000")
MODEL_SERVER_URL    = os.getenv("MODEL_SERVER_URL",   "http://model_server:8001")
PROMETHEUS_URL      = os.getenv("PROMETHEUS_URL",     "http://prometheus:9090")
SCRIPTS_DIR         = os.getenv("SCRIPTS_DIR",        "/opt/airflow/scripts")

# ── Absolute paths — NEVER use relative paths inside Airflow ──────────────────
APP_DIR      = "/app"
DATA_DIR     = "/app/data"
REPORTS_DIR  = "/app/reports"
MODELS_DIR   = "/app/models"
PARAMS_FILE  = "/app/params.yaml"

# ── Retraining thresholds ─────────────────────────────────────────────────────
RETRAIN_F1_DROP   = float(os.getenv("RETRAIN_F1_DROP_THRESHOLD", "0.02"))
DRIFT_THRESHOLD   = float(os.getenv("DRIFT_SCORE_THRESHOLD",     "0.15"))
MIN_F1_TO_PROMOTE = float(os.getenv("MIN_F1_TO_PROMOTE",          "0.02"))

# ── Ensure scripts dir is importable ─────────────────────────────────────────
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ── DAG default args ──────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "ddd_team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "start_date":       days_ago(1),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — called at the top of every task function
# ─────────────────────────────────────────────────────────────────────────────
def _setup_task_env():
    """
    Must be called at the start of every task function.
    - Changes CWD to /app so relative paths in train_*.py resolve correctly
    - Ensures writable temp dir exists for MLflow
    - Ensures output directories exist and are writable
    """
    os.chdir(APP_DIR)
    os.makedirs("/tmp/mlflow",  exist_ok=True)
    os.makedirs(REPORTS_DIR,    exist_ok=True)
    os.makedirs(MODELS_DIR,     exist_ok=True)

    # Re-assert env vars (Airflow workers can inherit a clean environment)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["GIT_PYTHON_REFRESH"]  = "quiet"

    # Override output paths used by train_xgboost.py and train_svm.py
    os.environ["XGB_MODEL_OUTPUT"]  = f"{MODELS_DIR}/xgboost_model.pkl"
    os.environ["SVM_MODEL_OUTPUT"]  = f"{MODELS_DIR}/svm_model.pkl"
    os.environ["XGB_METRICS_OUTPUT"]= f"{REPORTS_DIR}/xgboost_metrics.json"
    os.environ["SVM_METRICS_OUTPUT"]= f"{REPORTS_DIR}/svm_metrics.json"
    os.environ["TRAIN_CSV"]         = f"{DATA_DIR}/processed/train.csv"
    os.environ["TEST_CSV"]          = f"{DATA_DIR}/processed/test.csv"
    os.environ["PARAMS_FILE"]       = PARAMS_FILE

    # Make sure model_server scripts are importable
    model_server_path = "/app/model_server"
    if model_server_path not in sys.path:
        sys.path.insert(0, model_server_path)


# ─────────────────────────────────────────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def task_check_retrain_conditions(**context) -> bool:
    """T0: Gate — check whether retraining is actually needed."""
    _setup_task_env()
    ti      = context["ti"]
    conf    = context["dag_run"].conf or {}

    if conf.get("force_retrain", False):
        logger.info("force_retrain=True — skipping threshold checks")
        ti.xcom_push(key="retrain_reason", value="forced")
        return True

    reasons = []

    # Check Prometheus drift score
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "ddd_overall_drift_score"},
            timeout=5,
        )
        results = resp.json().get("data", {}).get("result", [])
        if results:
            drift_score = float(results[0]["value"][1])
            logger.info(f"Drift score: {drift_score}")
            if drift_score > DRIFT_THRESHOLD:
                reasons.append(f"drift_score={drift_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not fetch drift score: {e}")

    # Check Production model F1
    try:
        import mlflow
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_f1  = prod_run.data.metrics.get("f1_weighted", None)
            ti.xcom_push(key="prod_f1",      value=prod_f1)
            ti.xcom_push(key="prod_run_id",  value=prod_versions[0].run_id)
            ti.xcom_push(key="prod_version", value=prod_versions[0].version)
            logger.info(f"Production model F1: {prod_f1}")
        else:
            logger.warning("No Production model found — will retrain from scratch")
            ti.xcom_push(key="prod_f1", value=None)
            reasons.append("no_production_model")
    except Exception as e:
        logger.warning(f"Could not fetch MLflow production model: {e}")
        ti.xcom_push(key="prod_f1", value=None)
        reasons.append("mlflow_check_failed")

    if reasons:
        reason_str = ", ".join(reasons)
        logger.info(f"Retraining triggered: {reason_str}")
        ti.xcom_push(key="retrain_reason", value=reason_str)
        return True

    logger.info("Retraining conditions NOT met — skipping")
    return False


def task_pull_latest_data(**context):
    """T1: DVC pull latest dataset."""
    _setup_task_env()
    ti = context["ti"]

    logger.info("Pulling latest data from DVC remote...")
    try:
        result = subprocess.run(
            ["dvc", "pull", f"{DATA_DIR}/features/features.csv",
             f"{DATA_DIR}/processed/"],
            capture_output=True, text=True, timeout=300, cwd=APP_DIR,
        )
        status = "success" if result.returncode == 0 else "failed_using_local"
        logger.info(f"DVC pull: {status}")
        ti.xcom_push(key="dvc_pull_status", value=status)
    except Exception as e:
        logger.warning(f"DVC pull skipped: {e}")
        ti.xcom_push(key="dvc_pull_status", value="skipped")

    # Validate data exists
    features_path = f"{DATA_DIR}/features/features.csv"
    if not os.path.isfile(features_path):
        raise FileNotFoundError(
            f"features.csv not found at {features_path}. "
            "Run ddd_data_pipeline first."
        )

    import pandas as pd
    df = pd.read_csv(features_path)
    logger.info(f"Dataset ready: {len(df)} rows, "
                f"label dist: {df['label'].value_counts().to_dict()}")
    ti.xcom_push(key="n_samples", value=len(df))


def task_run_feature_engineering(**context):
    """T2: Re-run feature engineering only if explicitly requested."""
    _setup_task_env()
    ti   = context["ti"]
    conf = context["dag_run"].conf or {}

    if not conf.get("re_engineer_features", False):
        logger.info("Skipping feature re-engineering (use re_engineer_features=true to force)")
        ti.xcom_push(key="feature_eng_status", value="skipped")
        return {"status": "skipped"}

    from feature_engineering import run_feature_engineering
    from split_data import run_split

    summary = run_feature_engineering(
        landmarks_csv=f"{DATA_DIR}/landmarks/landmarks.csv",
        features_dir=f"{DATA_DIR}/features",
    )
    run_split(
        features_csv=f"{DATA_DIR}/features/features.csv",
        processed_dir=f"{DATA_DIR}/processed",
    )
    ti.xcom_push(key="feature_eng_status", value="completed")
    return {"status": "completed", **summary}


def task_train_xgboost(**context):
    """T3: Train new XGBoost model with full MLflow tracking."""
    _setup_task_env()   # ← sets CWD=/app and all env vars
    ti = context["ti"]

    logger.info(f"Training XGBoost (MLflow: {MLFLOW_TRACKING_URI}, "
                f"data: {os.environ['TRAIN_CSV']})")

    import yaml
    params = {}
    if os.path.isfile(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            params = yaml.safe_load(f)

    from train_xgboost import train
    metrics = train(params)

    logger.info(
        f"XGBoost done: F1={metrics.get('f1_weighted',0):.4f} "
        f"AUC={metrics.get('auc_roc',0):.4f} "
        f"run={metrics.get('mlflow_run_id','?')}"
    )
    ti.xcom_push(key="xgb_metrics",  value=metrics)
    ti.xcom_push(key="xgb_run_id",   value=metrics.get("mlflow_run_id"))
    ti.xcom_push(key="xgb_f1",       value=metrics.get("f1_weighted"))
    return metrics


def task_train_svm(**context):
    """T4: Train new SVM model with MLflow tracking."""
    _setup_task_env()
    ti = context["ti"]

    logger.info(f"Training SVM (MLflow: {MLFLOW_TRACKING_URI})")

    import yaml
    params = {}
    if os.path.isfile(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            params = yaml.safe_load(f)

    from train_svm import train
    metrics = train(params)

    logger.info(
        f"SVM done: F1={metrics.get('f1_weighted',0):.4f} "
        f"AUC={metrics.get('auc_roc',0):.4f}"
    )
    ti.xcom_push(key="svm_metrics", value=metrics)
    ti.xcom_push(key="svm_run_id",  value=metrics.get("mlflow_run_id"))
    ti.xcom_push(key="svm_f1",      value=metrics.get("f1_weighted"))
    return metrics


def task_evaluate_models(**context):
    """T5: Compare new models vs Production. Decide whether to promote."""
    _setup_task_env()
    ti = context["ti"]

    xgb_metrics = ti.xcom_pull(task_ids="t3_train_xgboost", key="xgb_metrics") or {}
    svm_metrics  = ti.xcom_pull(task_ids="t4_train_svm",     key="svm_metrics") or {}
    prod_f1      = ti.xcom_pull(task_ids="t0_check_retrain", key="prod_f1")

    xgb_f1 = float(xgb_metrics.get("f1_weighted", 0))
    svm_f1 = float(svm_metrics.get("f1_weighted", 0))

    if xgb_f1 >= svm_f1:
        winner, winner_f1, winner_run_id = "XGBoost", xgb_f1, \
            ti.xcom_pull(task_ids="t3_train_xgboost", key="xgb_run_id")
    else:
        winner, winner_f1, winner_run_id = "SVM", svm_f1, \
            ti.xcom_pull(task_ids="t4_train_svm", key="svm_run_id")

    prod_f1_val  = float(prod_f1) if prod_f1 is not None else 0.0
    improvement  = winner_f1 - prod_f1_val
    should_promote = (prod_f1 is None) or (improvement >= MIN_F1_TO_PROMOTE)

    comparison = {
        "xgb_f1": xgb_f1, "svm_f1": svm_f1,
        "winner": winner,  "winner_f1": winner_f1,
        "winner_run_id": winner_run_id,
        "prod_f1": prod_f1_val, "improvement": round(improvement, 4),
        "should_promote": should_promote,
    }

    logger.info(
        f"\n{'='*50}\nEVALUATION\n"
        f"  XGBoost F1    : {xgb_f1:.4f}\n"
        f"  SVM F1        : {svm_f1:.4f}\n"
        f"  Winner        : {winner}\n"
        f"  Production F1 : {prod_f1_val:.4f}\n"
        f"  Improvement   : {improvement:+.4f}\n"
        f"  Promote       : {should_promote}\n{'='*50}"
    )

    ti.xcom_push(key="comparison",    value=comparison)
    ti.xcom_push(key="should_promote",value=should_promote)
    ti.xcom_push(key="winner",        value=winner)
    ti.xcom_push(key="winner_run_id", value=winner_run_id)
    ti.xcom_push(key="winner_f1",     value=winner_f1)
    return comparison


def task_promote_if_better(**context) -> bool:
    """T6: Promote winner to MLflow Production stage if improvement warrants it."""
    _setup_task_env()
    ti         = context["ti"]
    comparison = ti.xcom_pull(task_ids="t5_evaluate_models", key="comparison") or {}

    should_promote = comparison.get("should_promote", False)
    winner         = comparison.get("winner", "unknown")
    winner_run_id  = comparison.get("winner_run_id")
    winner_f1      = comparison.get("winner_f1", 0)
    improvement    = comparison.get("improvement", 0)

    if not should_promote:
        logger.info(f"Promotion skipped: improvement={improvement:+.4f} < {MIN_F1_TO_PROMOTE}")
        ti.xcom_push(key="promoted", value=False)
        return False

    if not winner_run_id:
        logger.error("No winner_run_id — cannot promote")
        ti.xcom_push(key="promoted", value=False)
        return False

    import mlflow
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    all_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")

    target = next((v for v in all_versions if v.run_id == winner_run_id), None)
    if target is None:
        logger.warning(f"No MLflow version found for run_id={winner_run_id}. Skipping.")
        ti.xcom_push(key="promoted", value=False)
        return False

    for v in all_versions:
        if v.current_stage == "Production" and v.version != target.version:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME, version=v.version, stage="Archived"
            )

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME, version=target.version, stage="Production"
    )
    client.update_model_version(
        name=MLFLOW_MODEL_NAME,
        version=target.version,
        description=(
            f"Auto-promoted {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"{winner} | F1={winner_f1:.4f} | Δ={improvement:+.4f}"
        ),
    )

    logger.info(f"✓ Promoted {winner} v{target.version} → Production")
    ti.xcom_push(key="promoted",          value=True)
    ti.xcom_push(key="promoted_version",  value=target.version)
    return True


def task_reload_model_server(**context):
    """T7: Signal model server to hot-reload the new Production model."""
    _setup_task_env()
    ti       = context["ti"]
    promoted = ti.xcom_pull(task_ids="t6_promote_if_better", key="promoted")

    if not promoted:
        logger.info("Model not promoted — no reload needed")
        return {"status": "skipped"}

    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/admin/reload", timeout=30)
        if resp.status_code == 200:
            logger.info("✓ Model server reloaded")
            return {"status": "reloaded"}
        logger.warning(f"Reload returned {resp.status_code} — restart model_server manually")
    except Exception as e:
        logger.warning(f"Model server reload skipped: {e}")

    return {"status": "promoted_pending_restart",
            "note": "Run: docker compose restart model_server"}


def task_pipeline_report(**context):
    """T8: Save retraining report. Always runs (trigger_rule=ALL_DONE)."""
    _setup_task_env()
    ti = context["ti"]

    retrain_reason = ti.xcom_pull(task_ids="t0_check_retrain",    key="retrain_reason") or "unknown"
    n_samples      = ti.xcom_pull(task_ids="t1_pull_data",        key="n_samples") or 0
    xgb_metrics    = ti.xcom_pull(task_ids="t3_train_xgboost",    key="xgb_metrics") or {}
    svm_metrics    = ti.xcom_pull(task_ids="t4_train_svm",        key="svm_metrics") or {}
    comparison     = ti.xcom_pull(task_ids="t5_evaluate_models",  key="comparison") or {}
    promoted       = ti.xcom_pull(task_ids="t6_promote_if_better",key="promoted") or False
    promoted_ver   = ti.xcom_pull(task_ids="t6_promote_if_better",key="promoted_version")

    report = {
        "retrain_run": {
            "dag_run_id":     context["run_id"],
            "execution_date": str(context["execution_date"]),
            "retrain_reason": retrain_reason,
            "n_samples":      n_samples,
        },
        "new_models":  {"xgboost": xgb_metrics, "svm": svm_metrics},
        "comparison":  comparison,
        "promotion":   {"promoted": promoted, "promoted_version": promoted_ver},
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    report_path = os.path.join(
        REPORTS_DIR,
        f"retrain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    # DVC push new artifacts
    try:
        subprocess.run(
            ["dvc", "add",
             f"{MODELS_DIR}/xgboost_model.pkl",
             f"{MODELS_DIR}/svm_model.pkl"],
            capture_output=True, cwd=APP_DIR,
        )
        subprocess.run(["dvc", "push"], capture_output=True, cwd=APP_DIR)
        logger.info("DVC push complete")
    except Exception as e:
        logger.warning(f"DVC push skipped: {e}")

    logger.info(
        f"\n{'='*50}\nRETRAINING PIPELINE COMPLETE\n"
        f"  Reason   : {retrain_reason}\n"
        f"  Winner   : {comparison.get('winner','?')} "
        f"F1={comparison.get('winner_f1', 0):.4f}\n"
        f"  Promoted : {promoted}\n"
        f"  Report   : {report_path}\n{'='*50}"
    )

    ti.xcom_push(key="report_path", value=report_path)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddd_retrain_pipeline",
    description="Automated model retraining for the DDD system",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * 0",   # every Sunday at 2am
    catchup=False,
    max_active_runs=1,
    tags=["ddd", "retraining", "mlops", "automated"],
) as dag:

    t0_check = ShortCircuitOperator(
        task_id="t0_check_retrain",
        python_callable=task_check_retrain_conditions,
        provide_context=True,
    )
    t1_pull = PythonOperator(
        task_id="t1_pull_data",
        python_callable=task_pull_latest_data,
        provide_context=True,
    )
    t2_features = PythonOperator(
        task_id="t2_feature_engineering",
        python_callable=task_run_feature_engineering,
        provide_context=True,
    )
    t3_xgb = PythonOperator(
        task_id="t3_train_xgboost",
        python_callable=task_train_xgboost,
        provide_context=True,
    )
    t4_svm = PythonOperator(
        task_id="t4_train_svm",
        python_callable=task_train_svm,
        provide_context=True,
    )
    t5_eval = PythonOperator(
        task_id="t5_evaluate_models",
        python_callable=task_evaluate_models,
        provide_context=True,
    )
    t6_promote = ShortCircuitOperator(
        task_id="t6_promote_if_better",
        python_callable=task_promote_if_better,
        provide_context=True,
    )
    t7_reload = PythonOperator(
        task_id="t7_reload_model_server",
        python_callable=task_reload_model_server,
        provide_context=True,
    )
    t8_report = PythonOperator(
        task_id="t8_pipeline_report",
        python_callable=task_pipeline_report,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG chain
    t0_check >> t1_pull >> t2_features >> [t3_xgb, t4_svm]
    [t3_xgb, t4_svm] >> t5_eval >> t6_promote >> t7_reload >> t8_report
    t5_eval >> t8_report   # t8 always runs even if t6 short-circuits