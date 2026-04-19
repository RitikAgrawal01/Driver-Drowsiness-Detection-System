"""
backend/routers/stream.py
──────────────────────────
WebSocket /ws — real-time frame streaming endpoint.

Protocol (JSON messages):
  Client → Server:  {"type": "init",  "session_id": "...", "window_size": 30}
  Server → Client:  {"type": "init_ack", "session_id": "...", "status": "ready"}

  Client → Server:  {"type": "frame", "session_id": "...",
                      "frame_id": 42, "image_b64": "<base64-JPEG>"}
  Server → Client:  {"type": "prediction", "state": "alert",
                      "confidence": 0.92, "features": {...},
                      "alert_triggered": false, "inference_latency_ms": 18}

  Server → Client:  {"type": "alert", "severity": "warning",
                      "message": "Drowsiness detected!", "confidence": 0.89}

  Client → Server:  {"type": "close", "session_id": "..."}
  Server → Client:  {"type": "session_summary", ...}
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import get_settings
from backend.logger import get_logger
from backend.metrics import (
    ACTIVE_SESSIONS, DROWSY_ALERTS_TOTAL,
    FRAMES_PROCESSED_TOTAL, INFERENCE_LATENCY, PREDICTION_CONFIDENCE,
)
from backend.services import model_client
from backend.services.session_manager import get_session_manager

logger = get_logger("router.stream")
router = APIRouter(tags=["Streaming"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time drowsiness monitoring.

    Flow:
      1. Accept connection
      2. Wait for 'init' message with session_id
      3. Process 'frame' messages: extract features → infer → push result
      4. Push 'alert' messages when drowsy detected
      5. On 'close' message or disconnect: send session summary
    """
    await websocket.accept()
    settings = get_settings()
    manager = get_session_manager()

    session_id: Optional[str] = None
    session = None

    logger.info("WebSocket connection accepted")

    try:
        async for raw_msg in websocket.iter_text():
            # ── Parse message ─────────────────────────────────────────────────
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message",
                })
                continue

            msg_type = msg.get("type")

            # ── INIT ──────────────────────────────────────────────────────────
            if msg_type == "init":
                session_id = msg.get("session_id")
                window_size = int(msg.get("window_size",
                                          settings.sliding_window_size))

                # Create or attach to existing session
                if session_id:
                    session = manager.get_session(session_id)

                if session is None:
                    session = manager.create_session(window_size=window_size)
                    session_id = session.session_id

                logger.info(
                    f"WebSocket init: session={session_id} window={window_size}"
                )
                await websocket.send_json({
                    "type": "init_ack",
                    "session_id": session_id,
                    "status": "ready",
                    "window_size": window_size,
                })

            # ── FRAME ─────────────────────────────────────────────────────────
            elif msg_type == "frame":
                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Send 'init' message first",
                    })
                    continue

                frame_id = msg.get("frame_id", 0)
                image_b64 = msg.get("image_b64", "")

                session.frames_processed += 1
                FRAMES_PROCESSED_TOTAL.inc()

                t_start = time.perf_counter()

                # ── Feature extraction ────────────────────────────────────────
                feature_vector = session.extractor.process_frame(image_b64)

                if feature_vector is None:
                    # Window not full yet or no face detected
                    await websocket.send_json({
                        "type": "buffering",
                        "frame_id": frame_id,
                        "message": "Building feature window...",
                    })
                    continue

                # ── Model inference ───────────────────────────────────────────
                try:
                    result = await model_client.predict(feature_vector)
                except Exception as e:
                    logger.error(f"Model server error on frame {frame_id}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "frame_id": frame_id,
                        "message": "Model server temporarily unavailable",
                    })
                    continue

                latency_ms = (time.perf_counter() - t_start) * 1000
                state = result.get("state", "alert")
                confidence = result.get("confidence", 0.0)

                # Update session state
                session.current_state = state
                session.current_confidence = confidence

                ear = feature_vector.get("ear_mean")
                if ear is not None:
                    session.ear_history.append(ear)
                session.confidence_history.append(confidence)

                # Update drift detector
                session.drift_detector.update(feature_vector)

                # Prometheus
                INFERENCE_LATENCY.observe(latency_ms / 1000)
                PREDICTION_CONFIDENCE.observe(confidence)

                # ── Send prediction to client ─────────────────────────────────
                alert_triggered = (
                    state == "drowsy"
                    and confidence >= settings.confidence_threshold
                )

                await websocket.send_json({
                    "type": "prediction",
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "state": state,
                    "confidence": round(confidence, 4),
                    "features": {
                        k: round(v, 4) for k, v in feature_vector.items()
                    },
                    "alert_triggered": alert_triggered,
                    "inference_latency_ms": round(latency_ms, 2),
                })

                # ── Send drowsiness alert if triggered ────────────────────────
                if alert_triggered:
                    session.drowsy_alerts_triggered += 1
                    DROWSY_ALERTS_TOTAL.inc()
                    logger.warning(
                        f"DROWSY ALERT: session={session_id} "
                        f"frame={frame_id} confidence={confidence:.3f}"
                    )
                    await websocket.send_json({
                        "type": "alert",
                        "session_id": session_id,
                        "frame_id": frame_id,
                        "severity": "warning",
                        "message": "Drowsiness detected! Please take a break.",
                        "confidence": round(confidence, 4),
                    })

                # ── Periodic drift update (every 100 frames) ──────────────────
                if session.frames_processed % 100 == 0:
                    drift_scores = session.drift_detector.compute_drift_scores()
                    overall_drift = max(drift_scores.values(), default=0.0)
                    await websocket.send_json({
                        "type": "drift_update",
                        "overall_drift_score": round(overall_drift, 4),
                        "feature_scores": {
                            k: round(v, 4) for k, v in drift_scores.items()
                        },
                    })

            # ── CLOSE ─────────────────────────────────────────────────────────
            elif msg_type == "close":
                closed_session = manager.close_session(session_id)
                if closed_session:
                    summary = closed_session.to_summary_dict()
                    await websocket.send_json({
                        "type": "session_summary",
                        **summary,
                    })
                logger.info(f"WebSocket close requested: {session_id}")
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        # Clean up session on unexpected disconnect
        if session_id:
            manager.close_session(session_id)

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Internal server error",
            })
        except Exception:
            pass
        if session_id:
            manager.close_session(session_id)
