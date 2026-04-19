"""
backend/routers/session.py
───────────────────────────
Session management endpoints:
  POST /session/start
  POST /session/stop
  GET  /session/{session_id}/status
"""

from fastapi import APIRouter, HTTPException

from backend.logger import get_logger
from backend.schemas import (
    SessionStartRequest, SessionStartResponse,
    SessionStopRequest, SessionStopResponse,
    SessionStatusResponse,
)
from backend.services.session_manager import get_session_manager

logger = get_logger("router.session")
router = APIRouter(prefix="/session", tags=["Session"])


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Initialise a new drowsiness monitoring session.
    Returns a session_id used in all subsequent WebSocket and predict calls.
    """
    manager = get_session_manager()
    session = manager.create_session(
        window_size=request.window_size,
        driver_id=request.driver_id,
    )
    logger.info(f"Session started: {session.session_id}")
    return SessionStartResponse(
        session_id=session.session_id,
        started_at=session.started_at.isoformat(),
        window_size=session.window_size,
        status="active",
    )


@router.post("/stop", response_model=SessionStopResponse)
async def stop_session(request: SessionStopRequest):
    """
    Terminate an active session and return a summary report.
    """
    manager = get_session_manager()
    session = manager.close_session(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}",
        )

    summary = session.to_summary_dict()
    logger.info(
        f"Session stopped: {request.session_id} "
        f"({summary['total_frames_processed']} frames, "
        f"{summary['drowsy_alerts_triggered']} alerts)"
    )
    return SessionStopResponse(**summary)


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(session_id: str):
    """
    Return current real-time status of an active session.
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found or already closed: {session_id}",
        )

    return SessionStatusResponse(**session.to_status_dict())
