"""
airflow/dags/dag_data_pipeline.py
───────────────────────────────────
DAG: ddd_data_pipeline

End-to-end data engineering pipeline for the Driver Drowsiness Detection System.

Tasks (in order):
  t0_validate_raw_data      — Check raw video directory structure
  t1_extract_frames         — OpenCV: video → JPEG frames
  t2_extract_landmarks      — MediaPipe: frames → 468-landmark CSVs
  t3_feature_engineering    — PySpark: landmarks → EAR/MAR/PERCLOS/HeadPose features
  t4_split_data             — Stratified train/test split
  t5_dvc_add_and_push       — DVC version all outputs and push to remote
  t6_pipeline_summary       — Log final summary and throughput to Airflow XComs

Schedule: @once (triggered manually or by retraining DAG)
Can be triggered from Airflow UI or CLI:
  airflow dags trigger ddd_data_pipeline
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── DAG default args ──────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "ddd_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "start_date": days_ago(1),
}

# ── Paths (must match paths inside the Airflow container) ─────────────────────
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
LANDMARKS_CSV = os.path.join(LANDMARKS_DIR, "landmarks.csv")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
FEATURES_CSV = os.path.join(FEATURES_DIR, "features.csv")
BASELINE_JSON = os.path.join(FEATURES_DIR, "baseline.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/app/reports")

# ── Scripts dir ───────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.getenv("SCRIPTS_DIR", "/opt/airflow/dags/../scripts")

# ── Ensure scripts are importable ─────────────────────────────────────────────
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

logger = logging.getLogger("dag_data_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# TASK FUNCTIONS
# Each task function is wrapped by a PythonOperator below.
# They push results to XCom via the 'ti' (task instance) context.
# ─────────────────────────────────────────────────────────────────────────────

def task_validate_raw_data(**context):
    """
    T0: Validate raw data directory structure.
    Checks that data/raw/drowsy/ and data/raw/alert/ exist and contain videos.
    If using synthetic data, check that landmarks.csv already exists.
    """
    ti = context["ti"]

    # If synthetic landmark CSV already exists, skip raw data validation
    if os.path.isfile(LANDMARKS_CSV):
        msg = f"Synthetic/pre-computed landmarks found at {LANDMARKS_CSV} — skipping raw validation"
        logger.info(msg)
        ti.xcom_push(key="validation_status", value="skipped_synthetic")
        ti.xcom_push(key="use_synthetic", value=True)
        return {"status": "skipped", "reason": "synthetic_landmarks_exist"}

    ti.xcom_push(key="use_synthetic", value=False)

    issues = []
    for label in ["drowsy", "alert"]:
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            issues.append(f"Missing directory: {label_dir}")
            continue
        videos = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
        if not videos:
            issues.append(f"No video files in {label_dir}")
        else:
            logger.info(f"✓ Found {len(videos)} video(s) in {label_dir}")

    if issues:
        error_msg = "Raw data validation failed:\n" + "\n".join(issues)
        logger.error(error_msg)
        raise ValueError(error_msg)

    ti.xcom_push(key="validation_status", value="passed")
    return {"status": "passed", "issues": []}


def task_extract_frames(**context):
    """
    T1: Extract frames from raw videos using OpenCV.
    Skipped automatically if using synthetic landmark data.
    """
    ti = context["ti"]
    use_synthetic = ti.xcom_pull(task_ids="t0_validate_raw_data", key="use_synthetic")

    if use_synthetic:
        logger.info("Synthetic mode: skipping frame extraction")
        ti.xcom_push(key="extraction_summary", value={"status": "skipped"})
        return {"status": "skipped"}

    # Import here so Airflow worker can find it
    sys.path.insert(0, SCRIPTS_DIR)
    from extract_frames import run_extraction, validate_output

    summary = run_extraction(
        raw_dir=RAW_DIR,
        frames_dir=FRAMES_DIR,
        target_fps=int(os.getenv("FRAME_RATE", "30")),
    )

    if summary["total_frames_extracted"] == 0:
        raise ValueError("Frame extraction produced 0 frames!")

    if not validate_output(FRAMES_DIR):
        raise ValueError("Frame extraction output validation failed!")

    # Save summary to reports dir
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ti.xcom_push(key="extraction_summary", value=summary)

    logger.info(
        f"T1 complete: {summary['total_frames_extracted']} frames extracted "
        f"at {summary['overall_throughput_fps']} fps"
    )
    return summary


def task_extract_landmarks(**context):
    """
    T2: Run MediaPipe FaceMesh on extracted frames.
    Skipped if landmarks.csv already exists (synthetic mode).
    """
    ti = context["ti"]
    use_synthetic = ti.xcom_pull(task_ids="t0_validate_raw_data", key="use_synthetic")

    if use_synthetic:
        logger.info("Synthetic mode: landmarks.csv already present, skipping MediaPipe extraction")
        ti.xcom_push(key="landmark_summary", value={"status": "skipped_synthetic"})
        return {"status": "skipped"}

    from extract_landmarks import run_landmark_extraction, validate_output

    summary = run_landmark_extraction(
        frames_dir=FRAMES_DIR,
        output_csv=LANDMARKS_CSV,
    )

    if summary["total_frames_succeeded"] == 0:
        raise ValueError("No landmarks extracted — check that frames contain visible faces!")

    if not validate_output(LANDMARKS_CSV):
        raise ValueError("Landmark extraction output validation failed!")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "landmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ti.xcom_push(key="landmark_summary", value=summary)
    logger.info(
        f"T2 complete: {summary['total_frames_succeeded']} frames with landmarks "
        f"(detection rate: {summary['detection_rate_pct']}%)"
    )
    return summary


def task_feature_engineering(**context):
    """
    T3: PySpark sliding-window feature engineering.
    Computes EAR, MAR, PERCLOS, head pose over configurable window.
    """
    ti = context["ti"]

    from feature_engineering import run_feature_engineering, validate_output

    summary = run_feature_engineering(
        landmarks_csv=LANDMARKS_CSV,
        features_dir=FEATURES_DIR,
        window_size=int(os.getenv("SLIDING_WINDOW_SIZE", "30")),
    )

    if summary["n_output_windows"] == 0:
        raise ValueError("Feature engineering produced 0 windows!")

    if not validate_output(FEATURES_CSV, BASELINE_JSON):
        raise ValueError("Feature engineering output validation failed!")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "feature_engineering_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ti.xcom_push(key="feature_summary", value=summary)
    logger.info(
        f"T3 complete: {summary['n_output_windows']} windows, "
        f"label dist: {summary['label_distribution']}"
    )
    return summary


def task_split_data(**context):
    """
    T4: Stratified train/test split.
    """
    ti = context["ti"]

    from split_data import run_split

    summary = run_split(
        features_csv=FEATURES_CSV,
        processed_dir=PROCESSED_DIR,
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ti.xcom_push(key="split_summary", value=summary)
    logger.info(
        f"T4 complete: {summary['train_samples']} train, {summary['test_samples']} test"
    )
    return summary


def task_pipeline_summary(**context):
    """
    T6: Collect all XCom summaries and log overall pipeline stats.
    Pushes final summary to XCom for downstream DAGs (e.g. dag_retrain).
    """
    ti = context["ti"]

    extraction = ti.xcom_pull(task_ids="t1_extract_frames", key="extraction_summary") or {}
    landmarks = ti.xcom_pull(task_ids="t2_extract_landmarks", key="landmark_summary") or {}
    features = ti.xcom_pull(task_ids="t3_feature_engineering", key="feature_summary") or {}
    split = ti.xcom_pull(task_ids="t4_split_data", key="split_summary") or {}

    pipeline_summary = {
        "pipeline": "ddd_data_pipeline",
        "run_id": context["run_id"],
        "execution_date": str(context["execution_date"]),
        "stages": {
            "extraction": extraction,
            "landmarks": landmarks,
            "features": features,
            "split": split,
        },
        "final_train_samples": split.get("train_samples", 0),
        "final_test_samples": split.get("test_samples", 0),
        "label_distribution": features.get("label_distribution", {}),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "pipeline_summary.json"), "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    ti.xcom_push(key="pipeline_summary", value=pipeline_summary)

    logger.info(
        f"\n{'='*60}\n"
        f"FULL PIPELINE COMPLETE\n"
        f"  Train samples : {pipeline_summary['final_train_samples']}\n"
        f"  Test samples  : {pipeline_summary['final_test_samples']}\n"
        f"  Label dist.   : {pipeline_summary['label_distribution']}\n"
        f"  Run ID        : {pipeline_summary['run_id']}\n"
        f"{'='*60}"
    )
    return pipeline_summary


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddd_data_pipeline",
    description="Driver Drowsiness Detection — data engineering pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,   # triggered manually or by retraining DAG
    catchup=False,
    max_active_runs=1,
    tags=["ddd", "data", "mediapipe", "pyspark"],
) as dag:

    t0_validate = PythonOperator(
        task_id="t0_validate_raw_data",
        python_callable=task_validate_raw_data,
        provide_context=True,
        doc_md="""
        ## T0: Validate Raw Data
        Checks data/raw/drowsy/ and data/raw/alert/ contain video files.
        If synthetic landmarks.csv exists, marks downstream frame/landmark
        tasks as skipped.
        """,
    )

    t1_frames = PythonOperator(
        task_id="t1_extract_frames",
        python_callable=task_extract_frames,
        provide_context=True,
        doc_md="""
        ## T1: Extract Frames
        Reads MP4/AVI videos → extracts JPEG frames at TARGET_FPS using OpenCV.
        Saves to data/frames/{drowsy,alert}/.
        Logs throughput (frames/sec) to reports/extraction_summary.json.
        """,
    )

    t2_landmarks = PythonOperator(
        task_id="t2_extract_landmarks",
        python_callable=task_extract_landmarks,
        provide_context=True,
        doc_md="""
        ## T2: Extract Landmarks
        Runs MediaPipe FaceMesh (468 3D landmarks) on each frame.
        Saves data/landmarks/landmarks.csv.
        Logs detection rate to reports/landmark_summary.json.
        """,
    )

    t3_features = PythonOperator(
        task_id="t3_feature_engineering",
        python_callable=task_feature_engineering,
        provide_context=True,
        doc_md="""
        ## T3: Feature Engineering (PySpark)
        Computes sliding-window features: EAR mean/min/std, PERCLOS,
        MAR mean/max, head pitch/yaw/roll mean.
        Saves data/features/features.csv and baseline.json.
        """,
    )

    t4_split = PythonOperator(
        task_id="t4_split_data",
        python_callable=task_split_data,
        provide_context=True,
        doc_md="""
        ## T4: Train/Test Split
        Stratified 80/20 split on label column.
        Saves data/processed/train.csv and test.csv.
        """,
    )

    # DVC: version all pipeline outputs and push to remote
    t5_dvc = BashOperator(
        task_id="t5_dvc_add_and_push",
        bash_command="""
            set -e
            echo "Adding DVC-tracked outputs..."
            cd /app
            dvc add data/frames data/landmarks data/features data/processed || true
            git add data/frames.dvc data/landmarks.dvc data/features.dvc data/processed.dvc || true
            git commit -m "Pipeline run: ${AIRFLOW_CTX_DAG_RUN_ID:-manual}" --allow-empty || true
            echo "Pushing to DVC remote..."
            dvc push || echo "DVC push failed (remote may not be configured) — continuing"
            echo "DVC step complete"
        """,
        doc_md="""
        ## T5: DVC Version & Push
        Runs dvc add on all pipeline outputs, commits .dvc files to git,
        and pushes data/model artifacts to the DVC remote storage.
        """,
    )

    t6_summary = PythonOperator(
        task_id="t6_pipeline_summary",
        python_callable=task_pipeline_summary,
        provide_context=True,
        doc_md="""
        ## T6: Pipeline Summary
        Aggregates XCom results from all tasks.
        Saves reports/pipeline_summary.json.
        """,
    )

    # ── DAG dependency chain ──────────────────────────────────────────────────
    # t0 → t1 → t2 → t3 → t4 → t5 → t6
    t0_validate >> t1_frames >> t2_landmarks >> t3_features >> t4_split >> t5_dvc >> t6_summary
