"""
backend/services/session_manager.py
─────────────────────────────────────
In-memory session state manager.
Tracks all active monitoring sessions and their statistics.
Thread-safe using asyncio locks.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from backend.logger import get_logger
from backend.metrics import ACTIVE_SESSIONS
from backend.services.feature_extractor import FeatureExtractor
from backend.services.drift_detector import DriftDetector
from backend.config import get_settings

logger = get_logger("session_manager")


class SessionState:
    """All state for one active monitoring session."""

    def __init__(self, session_id: str, window_size: int,
                 driver_id: Optional[str] = None):
        self.session_id = session_id
        self.driver_id = driver_id
        self.window_size = window_size
        self.started_at = datetime.now(timezone.utc)
        self.ended_at: Optional[datetime] = None
        self.status = "active"

        # Stats
        self.frames_processed = 0
        self.drowsy_alerts_triggered = 0
        self.ear_history: list = []
        self.confidence_history: list = []

        # Current values (updated per-frame)
        self.current_state: Optional[str] = None
        self.current_confidence: Optional[float] = None

        # To 
        self.drowsy_buffer = 0
        self.alert_buffer = 0
        self.current_state = "alert"

        # Sub-services (one per session)
        settings = get_settings()
        self.extractor = FeatureExtractor(
            window_size=window_size,
            ear_threshold=settings.ear_threshold,
            mar_threshold=settings.mar_threshold,
        )
        self.drift_detector = DriftDetector(
            window_size=settings.drift_window_size
        )

    def to_status_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "frames_processed": self.frames_processed,
            "current_state": self.current_state,
            "current_confidence": self.current_confidence,
            "current_ear": self.extractor.get_current_ear(),
            "current_mar": self.extractor.get_current_mar(),
            "current_perclos": self.extractor.get_current_perclos(),
            "alerts_triggered": self.drowsy_alerts_triggered,
            "drift_score": self.drift_detector.get_overall_score(),
        }

    def to_summary_dict(self) -> dict:
        ended = self.ended_at or datetime.now(timezone.utc)
        duration = (ended - self.started_at).total_seconds()
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": round(duration, 2),
            "total_frames_processed": self.frames_processed,
            "drowsy_alerts_triggered": self.drowsy_alerts_triggered,
            "average_ear": (
                round(sum(self.ear_history) / len(self.ear_history), 4)
                if self.ear_history else None
            ),
            "average_confidence": (
                round(sum(self.confidence_history) / len(self.confidence_history), 4)
                if self.confidence_history else None
            ),
            "status": "completed",
        }

    def cleanup(self):
        """Release MediaPipe and other resources."""
        try:
            self.extractor.close()
        except Exception:
            pass


class SessionManager:
    """
    Manages all active sessions.
    Singleton — instantiated once in main.py lifespan.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        logger.info("SessionManager initialised")

    def create_session(self, window_size: int,
                       driver_id: Optional[str] = None) -> SessionState:
        """Create and register a new session."""
        session_id = str(uuid.uuid4())
        session = SessionState(
            session_id=session_id,
            window_size=window_size,
            driver_id=driver_id,
        )
        self._sessions[session_id] = session
        ACTIVE_SESSIONS.inc()
        logger.info(
            f"Session created: {session_id} "
            f"(driver={driver_id}, window={window_size})"
        )
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> Optional[SessionState]:
        """Mark a session as completed and release its resources."""
        session = self._sessions.pop(session_id, None)
        if session:
            session.ended_at = datetime.now(timezone.utc)
            session.status = "completed"
            session.cleanup()
            ACTIVE_SESSIONS.dec()
            logger.info(
                f"Session closed: {session_id} "
                f"(frames={session.frames_processed}, "
                f"alerts={session.drowsy_alerts_triggered})"
            )
        return session

    def list_sessions(self) -> list:
        return [
            {"session_id": sid, "status": s.status,
             "frames": s.frames_processed}
            for sid, s in self._sessions.items()
        ]

    def active_count(self) -> int:
        return len(self._sessions)

    def cleanup_all(self):
        """Release all sessions (called on app shutdown)."""
        for session in self._sessions.values():
            session.cleanup()
        self._sessions.clear()


# ── Singleton instance (set in main.py lifespan) ──────────────────────────────
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
