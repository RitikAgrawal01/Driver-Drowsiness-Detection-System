"""
ddd_data_pipeline — validate raw videos, then build features with DVC (manual trigger).

  t0_validate_raw_data   names, alert/drowsy pairs, duplicate-content check (v1 shipped a duplicate)
  t1_dvc_pull            fetch cached outputs from the DVC remote
  t2_dvc_repro_features  dvc repro window_features  (landmarks -> frame features -> windows)
  t3_dvc_push            store the new outputs in the remote
  t4_summary             log the window summary

Every step fails loudly; there is no `|| true`. What runs, and with which settings, is defined
once in dvc.yaml + params.yaml — Airflow only decides when.
"""

import json
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG
from ddd_dag_utils import REPO, sh, validate_raw_videos

default_args = {"owner": "ddd", "retries": 0, "retry_delay": timedelta(minutes=2)}


def summary():
    print(json.dumps(json.loads((REPO / "reports" / "window_summary.json").read_text())["per_window_length"],
                     indent=2))


with DAG(
    dag_id="ddd_data_pipeline",
    description="Raw videos -> windows via dvc repro",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["ddd", "dvc", "data"],
) as dag:
    t0 = PythonOperator(task_id="t0_validate_raw_data",
                        python_callable=lambda: print(validate_raw_videos(REPO / "data" / "raw")))
    t1 = PythonOperator(task_id="t1_dvc_pull", python_callable=lambda: sh("dvc pull --allow-missing"),
                        retries=2)
    t2 = PythonOperator(task_id="t2_dvc_repro_features", python_callable=lambda: sh("dvc repro window_features"),
                        execution_timeout=timedelta(hours=3))
    t3 = PythonOperator(task_id="t3_dvc_push", python_callable=lambda: sh("dvc push"), retries=2)
    t4 = PythonOperator(task_id="t4_summary", python_callable=summary)
    t0 >> t1 >> t2 >> t3 >> t4
