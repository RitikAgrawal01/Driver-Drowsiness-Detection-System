"""
Read-only model/experiment info for the UI. Replaces v1's open proxy
(POST /api/mlflow/runs/search forwarded ANY body to MLflow, unauthenticated, with a blocking
`requests` call inside an async route).
"""

import httpx
from fastapi import APIRouter, HTTPException, Query

from backend.config import get_settings
from backend.services.drift_monitor import drift_monitor
from backend.services.serving import serving
from ddd_core.config import load_params

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/model")
async def current_model():
    await serving.refresh()
    if serving.model is None:
        raise HTTPException(status_code=503, detail="model server not ready")
    return {**serving.model, "config_error": serving.config_error}


@router.get("/runs")
async def recent_runs(limit: int = Query(20, ge=1, le=50)):
    """Latest runs of the configured experiment only (fixed query, capped size)."""
    exp_name = load_params()["mlflow"]["experiment"]
    base = get_settings().mlflow_tracking_uri.rstrip("/")
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{base}/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": exp_name})
            if r.status_code != 200:
                raise HTTPException(status_code=404, detail=f"experiment {exp_name} not found")
            exp_id = r.json()["experiment"]["experiment_id"]
            r = await c.post(
                f"{base}/api/2.0/mlflow/runs/search",
                json={"experiment_ids": [exp_id], "max_results": limit, "order_by": ["attributes.start_time DESC"]},
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"MLflow unreachable: {e}")
    runs = []
    for run in r.json().get("runs", []):
        info, data = run["info"], run.get("data", {})
        runs.append(
            {
                "run_id": info["run_id"],
                "run_name": info.get("run_name"),
                "status": info.get("status"),
                "start_time": info.get("start_time"),
                "metrics": {m["key"]: m["value"] for m in data.get("metrics", [])},
                "params": {p["key"]: p["value"] for p in data.get("params", [])},
                "tags": {t["key"]: t["value"] for t in data.get("tags", []) if not t["key"].startswith("mlflow.")},
            }
        )
    return {"experiment": exp_name, "runs": runs}


@router.get("/drift")
async def drift():
    threshold = load_params()["drift"]["psi_threshold"]
    return {
        **(drift_monitor.scores or {"overall": -1, "note": "not enough live windows yet"}),
        "threshold": threshold,
        "method": "psi",
    }


PIPELINE_DAGS = {"ddd_data_pipeline", "ddd_retrain_pipeline"}


async def _airflow_get(path: str, params: dict | None = None) -> dict:
    s = get_settings()
    if not s.airflow_user:
        raise HTTPException(status_code=503, detail="Airflow credentials not configured")
    async with httpx.AsyncClient(timeout=5.0, auth=(s.airflow_user, s.airflow_password)) as c:
        try:
            r = await c.get(f"{s.airflow_url.rstrip('/')}/api/v1{path}", params=params)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Airflow unreachable: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:200])
    return r.json()


@router.get("/pipeline/{dag_id}/runs")
async def dag_runs(dag_id: str, limit: int = Query(10, ge=1, le=25)):
    if dag_id not in PIPELINE_DAGS:
        raise HTTPException(status_code=404, detail="unknown DAG")
    return await _airflow_get(f"/dags/{dag_id}/dagRuns", {"limit": limit, "order_by": "-execution_date"})


@router.get("/pipeline/{dag_id}/runs/{run_id}/tasks")
async def dag_run_tasks(dag_id: str, run_id: str):
    if dag_id not in PIPELINE_DAGS:
        raise HTTPException(status_code=404, detail="unknown DAG")
    return await _airflow_get(f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
