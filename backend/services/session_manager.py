"""
Monitoring sessions (in memory, one per WebSocket client).

handle_frame() is the whole per-frame flow, kept out of the WebSocket router so it can be
unit-tested with a fake extractor and a fake model client:

  frame -> size check -> timestamp sampling (10 fps) -> MediaPipe in a worker thread
        -> every `stride_s` seconds once the window is full: window features -> model server
        -> alarm state machine (seconds) -> drift buffer -> status message

Displayed states (fail-safe: an error or a missing face is never shown as "alert"):
  warming_up  window not yet full
  no_face     face missing for >= serving.no_face_warn_s
  error       the last model call failed (model server down / 5xx / timeout)
  degraded    measured input fps below serving.min_fps
  drowsy      alarm on (drowsy predictions sustained >= alarm_on_s)
  alert       otherwise
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone

from backend.logger import get_logger
from backend.metrics import (
    ACTIVE_SESSIONS,
    DROWSY_ALARMS,
    FEATURE_EXTRACTION_LATENCY,
    FRAMES_PROCESSED,
    FRAMES_SKIPPED,
    PREDICTION_ERRORS,
    PREDICTION_SCORE,
    PREDICTIONS,
    STATE_TRANSITIONS,
)
from backend.services import model_client
from backend.services.drift_monitor import drift_monitor
from backend.services.live_features import FrameDecodeError, LiveFeatureExtractor, decode_frame
from ddd_core.alarm import AlarmStateMachine
from ddd_core.config import load_params

logger = get_logger("session_manager")


class MonitoringSession:
    def __init__(self, window_s: float, model_version: str | None, extractor=None, predictor=None):
        p = load_params()
        self.serving, feat_cfg = p["serving"], p["features"]
        self.stride_s = float(feat_cfg["stride_s"])
        self.session_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.model_version = model_version
        self.extractor = extractor or LiveFeatureExtractor(
            window_s, self.serving["target_fps"], self.serving["max_side"], feat_cfg
        )
        self.predict = predictor or model_client.predict
        self.alarm = AlarmStateMachine(self.serving["alarm_on_s"], self.serving["alarm_off_s"])
        self.state = "warming_up"
        self.last_prediction_t: float | None = None
        self.last_result: dict | None = None
        self.last_features: dict | None = None
        self.model_error: str | None = None
        self.low_quality = False
        self.frames_received = self.frames_processed = self.predictions = 0
        self.alarms = 0
        self.state_seconds: dict = {}
        self._last_state_t: float | None = None
        self._t_offset: float | None = None

    # ── time ───────────────────────────────────────────────────────────────
    def _stream_time(self, ts_ms) -> float:
        """Client timestamp (ms) -> seconds; falls back to server time. Non-monotonic -> server time."""
        if isinstance(ts_ms, (int, float)) and ts_ms > 0:
            t = ts_ms / 1000.0
            if self.extractor.frames and t < self.extractor.frames[-1][0]:
                return self.extractor.frames[-1][0] + 1e-3
            return t
        if self._t_offset is None:
            self._t_offset = time.monotonic()
        return time.monotonic() - self._t_offset

    def _set_state(self, new: str, t: float) -> None:
        if self._last_state_t is not None:
            self.state_seconds[self.state] = self.state_seconds.get(self.state, 0.0) + (t - self._last_state_t)
        self._last_state_t = t
        if new != self.state:
            STATE_TRANSITIONS.labels(to_state=new).inc()
            self.state = new

    # ── per frame ──────────────────────────────────────────────────────────
    async def handle_frame(self, image_b64: str, ts_ms=None) -> dict:
        self.frames_received += 1
        if len(image_b64) > self.serving["max_frame_bytes"]:
            FRAMES_SKIPPED.labels(reason="too_large").inc()
            return {"type": "error", "message": "frame too large"}
        t = self._stream_time(ts_ms)
        if not self.extractor.should_sample(t):
            FRAMES_SKIPPED.labels(reason="rate_limit").inc()
            return self._status(t, processed=False)
        try:
            img = decode_frame(image_b64)
        except FrameDecodeError as e:
            FRAMES_SKIPPED.labels(reason="decode_error").inc()
            return {"type": "error", "message": str(e)}

        t0 = time.perf_counter()
        await asyncio.to_thread(self.extractor.process_frame, img, t)  # keep the event loop free
        FEATURE_EXTRACTION_LATENCY.observe(time.perf_counter() - t0)
        FRAMES_PROCESSED.inc()
        self.frames_processed += 1

        new_alarm, predicted = False, False
        due = self.last_prediction_t is None or t - self.last_prediction_t >= self.stride_s - 1e-6
        if self.extractor.window_ready(t) and due:
            window = self.extractor.compute_window(t)
            self.last_prediction_t = t
            self.low_quality = window is None  # face seen in < min_face_ratio of the window
            if window is not None:
                new_alarm = await self._predict(window, t)
                predicted = self.model_error is None
        msg = self._status(t, processed=True, new_alarm=new_alarm)
        msg["predicted"] = predicted
        return msg

    async def _predict(self, window: dict, t: float) -> bool:
        feats = {k: v for k, v in window.items() if k != "face_ratio"}
        PREDICTIONS.labels(source="stream").inc()
        try:
            result = await self.predict(feats)
        except model_client.ModelServerError as e:
            PREDICTION_ERRORS.labels(source="stream", error_type=e.error_type).inc()
            self.model_error = e.error_type
            logger.error("model call failed", session=self.session_id, error=e.error_type)
            return False
        except Exception as e:  # anything unexpected is still an error, never "alert"
            PREDICTION_ERRORS.labels(source="stream", error_type="unexpected").inc()
            self.model_error = "unexpected"
            logger.exception("model call failed", session=self.session_id, error=str(e))
            return False
        self.model_error = None
        self.predictions += 1
        self.last_result, self.last_features = result, feats
        PREDICTION_SCORE.observe(result["score"])
        drift_monitor.add(feats)
        if self.alarm.update(t, result["state"] == "drowsy"):
            self.alarms += 1
            DROWSY_ALARMS.inc()
            return True
        return False

    def _status(self, t: float, processed: bool, new_alarm: bool = False) -> dict:
        ex = self.extractor
        if not ex.window_ready(t):
            state = "warming_up"
        elif ex.seconds_without_face(t) >= self.serving["no_face_warn_s"]:
            state = "no_face"
        elif self.model_error:
            state = "error"
        elif ex.measured_fps() < self.serving["min_fps"] or self.low_quality:
            state = "degraded"
        elif self.last_result is None:
            state = "warming_up"
        else:
            state = "drowsy" if self.alarm.alarm else "alert"
        self._set_state(state, t)
        msg = {
            "type": "status",
            "session_id": self.session_id,
            "state": state,
            "alarm": self.alarm.alarm,
            "new_alarm": new_alarm,
            "processed": processed,
            "warmup": round(ex.warmup_progress(t), 3),
            "fps": round(ex.measured_fps(), 1),
            "seconds_without_face": round(ex.seconds_without_face(t), 1),
        }
        if self.model_error:
            msg["error"] = self.model_error
        if self.last_result is not None:
            msg.update(
                score=self.last_result["score"],
                threshold=self.last_result["threshold"],
                raw_prediction=self.last_result["state"],
                model_version=self.last_result["model_version"],
                features={k: round(float(v), 4) for k, v in self.last_features.items()},
            )
        return msg

    def summary(self) -> dict:
        ended = datetime.now(timezone.utc)
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": round((ended - self.started_at).total_seconds(), 2),
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "predictions": self.predictions,
            "drowsy_alarms": self.alarms,
            "seconds_per_state": {k: round(v, 1) for k, v in self.state_seconds.items()},
            "model_version": self.model_version,
        }

    def close(self) -> None:
        self.extractor.close()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, MonitoringSession] = {}

    def create(self, window_s: float, model_version: str | None, **kw) -> MonitoringSession:
        s = MonitoringSession(window_s, model_version, **kw)
        self._sessions[s.session_id] = s
        ACTIVE_SESSIONS.set(len(self._sessions))
        logger.info("session created", session=s.session_id, window_s=window_s, model_version=model_version)
        return s

    def get(self, session_id: str) -> MonitoringSession | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str | None) -> MonitoringSession | None:
        s = self._sessions.pop(session_id, None) if session_id else None
        if s is not None:
            s.close()
            ACTIVE_SESSIONS.set(len(self._sessions))
            logger.info("session closed", session=session_id, frames=s.frames_processed, alarms=s.alarms)
        return s

    def active_count(self) -> int:
        return len(self._sessions)

    def close_all(self) -> None:
        for sid in list(self._sessions):
            self.close(sid)


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
