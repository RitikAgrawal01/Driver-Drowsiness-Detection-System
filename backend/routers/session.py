"""Read/stop endpoints for live sessions (sessions are created by the WebSocket 'init')."""

from fastapi import APIRouter, HTTPException

from backend.services.session_manager import get_session_manager

router = APIRouter(prefix="/session", tags=["Session"])


@router.get("/{session_id}/status")
async def session_status(session_id: str):
    s = get_session_manager().get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found or closed")
    return {
        "session_id": s.session_id,
        "state": s.state,
        "alarm": s.alarm.alarm,
        "frames_processed": s.frames_processed,
        "predictions": s.predictions,
        "drowsy_alarms": s.alarms,
        "model_version": s.model_version,
    }


@router.post("/{session_id}/stop")
async def stop_session(session_id: str):
    s = get_session_manager().close(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found or closed")
    return s.summary()
