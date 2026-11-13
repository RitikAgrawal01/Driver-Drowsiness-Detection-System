"""
ddd_retrain_pipeline — retrain, gate and promote (weekly, manual, or triggered by a drift alert
through Alertmanager -> webhook).

  t0_check_trigger        ShortCircuit: alert/manual runs proceed; scheduled runs only if
                          `dvc status` shows something changed (new data or params)
  t1_dvc_pull             fetch inputs/cached outputs
  t2_dvc_repro            the whole DVC pipeline: features -> nested LOSO -> GATE -> train_final.
                          The gate stage exits 1 when nothing may be promoted, which fails this
                          task, so nothing below runs. Promotion = MLflow alias @champion (set
                          inside train_final); there is no separate "promote" logic to drift.
  t3_dvc_push             store outputs
  t4_reload_model_server  POST /admin/reload; fails if the new champion does not load
  t5_verify_serving       model server serves the version train_final registered; backend ready
  t6_report               ALWAYS runs (trigger_rule=all_done), also after a short-circuit or failure

v1 fixes: no hand-rolled training/promotion code (it duplicated DVC with different settings);
no `|| true`; the ShortCircuit no longer skips the report (ignore_downstream_trigger_rules=False);
"MLflow unreachable" can no longer promote (the gate fails closed).
"""

import json
import os
import urllib.request
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG
from ddd_dag_utils import REPO, sh, should_retrain, stale_stages

MODEL_SERVER = os.getenv("MODEL_SERVER_URL", "http://model_server:8001")
BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")


def check_trigger(**ctx) -> bool:
    run = ctx["dag_run"]
    ok, reason = should_retrain(run.conf, run.run_type, stale_stages())
    print(reason)
    ctx["ti"].xcom_push(key="reason", value=reason)
    return ok


def http_json(url: str, method: str = "GET", headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=60) as r:  # raises on 4xx/5xx -> task fails
        return json.loads(r.read() or b"{}")


def reload_model_server():
    headers = {"X-Admin-Token": os.environ["ADMIN_TOKEN"]} if os.getenv("ADMIN_TOKEN") else {}
    print(http_json(f"{MODEL_SERVER}/admin/reload", "POST", headers))


def verify_serving():
    final = json.loads((REPO / "reports" / "final_model.json").read_text())
    info = http_json(f"{MODEL_SERVER}/model/info")
    if str(info["model_version"]) != str(final["version"]):
        raise RuntimeError(f"model server serves v{info['model_version']}, expected v{final['version']}")
    print("model server OK:", info["model_version"], info["model_type"])
    print("backend ready:", http_json(f"{BACKEND}/ready"))


def report(**ctx):
    run = ctx["dag_run"]
    states = {ti.task_id: ti.state for ti in run.get_task_instances()}
    gate_path = REPO / "reports" / "gate.json"
    out = {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "conf": run.conf,
        "reason": ctx["ti"].xcom_pull(task_ids="t0_check_trigger", key="reason"),
        "task_states": states,
        "gate": json.loads(gate_path.read_text()) if gate_path.exists() else None,
    }
    target = REPO / "logs" / "retrain_runs"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{run.run_id.replace(':', '_')}.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


with DAG(
    dag_id="ddd_retrain_pipeline",
    description="dvc repro (LOSO + gate + @champion) -> hot reload -> verify",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ddd", "retries": 0},
    tags=["ddd", "dvc", "mlflow", "retrain"],
) as dag:
    t0 = ShortCircuitOperator(
        task_id="t0_check_trigger", python_callable=check_trigger, ignore_downstream_trigger_rules=False
    )
    t1 = PythonOperator(task_id="t1_dvc_pull", python_callable=lambda: sh("dvc pull --allow-missing"), retries=2)
    t2 = PythonOperator(
        task_id="t2_dvc_repro", python_callable=lambda: sh("dvc repro"), execution_timeout=timedelta(hours=4)
    )
    t3 = PythonOperator(task_id="t3_dvc_push", python_callable=lambda: sh("dvc push"), retries=2)
    t4 = PythonOperator(task_id="t4_reload_model_server", python_callable=reload_model_server, retries=2)
    t5 = PythonOperator(
        task_id="t5_verify_serving", python_callable=verify_serving, retries=2, retry_delay=timedelta(seconds=40)
    )
    t6 = PythonOperator(task_id="t6_report", python_callable=report, trigger_rule=TriggerRule.ALL_DONE)
    t0 >> t1 >> t2 >> t3 >> t4 >> t5 >> t6
