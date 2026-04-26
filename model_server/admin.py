"""
model_server/admin.py
──────────────────────
Admin endpoints for the model server.
Adds a POST /admin/reload endpoint that lets the retraining DAG
hot-swap the Production model without restarting the container.

Registered as a router in model_server/main.py.
"""

import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
import mlflow

from model_loader import load_model, get_model_meta, is_loaded

logger = logging.getLogger("model_server.admin")
router = APIRouter(prefix="/admin", tags=["Admin"])

# Reload lock — prevent concurrent reloads
_reloading = False


async def _do_reload():
    """Background task: reload model from MLflow registry."""
    global _reloading
    _reloading = True
    try:
        logger.info("Hot-reloading Production model from MLflow registry...")
        success = load_model()
        if success:
            meta = get_model_meta()
            logger.info(
                f"✓ Model hot-reloaded: {meta['algorithm']} "
                f"v{meta['model_version']} from {meta['source']}"
            )
        else:
            logger.error("Hot-reload failed — model server still running previous model")
    except Exception as e:
        logger.error(f"Hot-reload exception: {e}")
    finally:
        _reloading = False


@router.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Hot-reload the Production model from MLflow registry.
    Called by dag_retrain after promoting a new model.
    The reload happens in the background so this endpoint
    returns immediately (non-blocking).
    """
    global _reloading

    if _reloading:
        raise HTTPException(
            status_code=409,
            detail="Model reload already in progress. Try again in 30 seconds.",
        )

    background_tasks.add_task(_do_reload)

    return {
        "status": "reload_started",
        "message": "New Production model is being loaded in the background.",
        "current_model": get_model_meta(),
    }


@router.get("/status")
async def admin_status():
    """Returns admin status including reload state."""
    return {
        "model_loaded": is_loaded(),
        "reload_in_progress": _reloading,
        "model_meta": get_model_meta(),
    }
