"""
WebSocket /ws — real-time monitoring.

Protocol (JSON):
  C->S {"type": "init"}                                   (client settings are NOT trusted)
  S->C {"type": "init_ack", "session_id", "window_s", "target_fps", "model_version"}
  C->S {"type": "frame", "ts": <ms, monotonic>, "image_b64": "<JPEG>"}
  S->C {"type": "status", "state": warming_up|alert|drowsy|no_face|degraded|error, "alarm",
        "new_alarm", "score", "threshold", "features", "fps", "warmup", ...}   one reply per frame
  C->S {"type": "close"}
  S->C {"type": "session_summary", ...}

Backpressure: the server answers every frame and the client keeps at most one frame in
flight, so frames never queue up behind a slow server.
"""

import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.logger import get_logger
from backend.metrics import FRAME_LATENCY, FRAMES_RECEIVED, INFERENCE_LATENCY
from backend.services.serving import serving
from backend.services.session_manager import get_session_manager
from ddd_core.config import load_params

logger = get_logger("router.stream")
router = APIRouter(tags=["Streaming"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager = get_session_manager()
    session = None
    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "invalid JSON"})
                continue
            kind = msg.get("type")

            if kind == "init":
                if not serving.ready:
                    await serving.refresh()
                if not serving.ready:
                    await websocket.send_json(
                        {"type": "error", "state": "error", "message": serving.config_error or "model not available"}
                    )
                    continue
                if session is not None:
                    manager.close(session.session_id)
                m = serving.model
                session = manager.create(window_s=float(m["window_s"]), model_version=m["model_version"])
                await websocket.send_json(
                    {
                        "type": "init_ack",
                        "session_id": session.session_id,
                        "window_s": m["window_s"],
                        "target_fps": load_params()["serving"]["target_fps"],
                        "model_version": m["model_version"],
                    }
                )

            elif kind == "frame":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "send 'init' first"})
                    continue
                FRAMES_RECEIVED.inc()
                t0 = time.perf_counter()
                out = await session.handle_frame(msg.get("image_b64", ""), msg.get("ts"))
                await websocket.send_json(out)
                dt = time.perf_counter() - t0
                FRAME_LATENCY.observe(dt)
                if out.get("predicted"):
                    INFERENCE_LATENCY.observe(dt)

            elif kind == "close":
                if session is not None:
                    await websocket.send_json({"type": "session_summary", **session.summary()})
                    manager.close(session.session_id)
                    session = None
                break

            else:
                await websocket.send_json({"type": "error", "message": f"unknown message type: {kind}"})

    except WebSocketDisconnect:
        logger.info("websocket disconnected")
    except Exception as e:
        logger.exception("websocket error", error=str(e))
        try:
            await websocket.send_json({"type": "error", "state": "error", "message": "internal server error"})
        except Exception:
            pass
    finally:
        if session is not None:
            manager.close(session.session_id)
