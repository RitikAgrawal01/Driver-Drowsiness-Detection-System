"""Admin endpoints: POST /admin/reload hot-swaps to the current @champion (called by the retrain DAG)."""

import asyncio
import hmac
import os

from fastapi import APIRouter, Header, HTTPException

router = APIRouter(prefix="/admin", tags=["Admin"])
_reload_lock = asyncio.Lock()


def _check_token(token: str | None) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if expected and not (token and hmac.compare_digest(token, expected)):
        raise HTTPException(status_code=401, detail="Invalid admin token")


@router.post("/reload")
async def reload_model(x_admin_token: str | None = Header(default=None)):
    """Synchronous reload: returns the new model's metadata, or 500 and keeps the old model."""
    _check_token(x_admin_token)
    from model_server.main import try_load
    from model_server.model_loader import store

    if _reload_lock.locked():
        raise HTTPException(status_code=409, detail="Reload already in progress")
    async with _reload_lock:
        ok = await asyncio.to_thread(try_load)
    if not ok:
        raise HTTPException(status_code=500, detail="Reload failed; previous model (if any) still serving")
    return {"status": "reloaded", "model": store.meta()}


@router.get("/status")
async def admin_status():
    from model_server.model_loader import store

    return {"model_loaded": store.is_loaded(), "reload_in_progress": _reload_lock.locked(), "model": store.meta()}
