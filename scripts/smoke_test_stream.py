"""
End-to-end smoke test of the running stack: stream a recorded video through the backend
WebSocket exactly like the browser does (10 fps, timestamped JPEGs, one frame in flight) and
summarise the states that come back.

  python scripts/smoke_test_stream.py data/raw/drowsy/drowsy3.mov --seconds 90
Exit code 1 if no prediction arrives or any 'error' state is returned.
"""

import argparse
import asyncio
import base64
import collections
import json
import sys
import time

import cv2
import websockets

ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def frames(path: str, fps: float, seconds: float):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step, i, next_t = 1 / fps, -1, 0.0
    while True:
        ok = cap.grab()
        i += 1
        t = i / src
        if not ok or t > seconds:
            break
        if t + 1e-6 < next_t:
            continue
        next_t += step
        ok, img = cap.retrieve()
        if not ok:
            continue
        if rot in ROTATE:
            img = cv2.rotate(img, ROTATE[rot])
        h, w = img.shape[:2]
        s = min(1.0, 640 / w)
        img = cv2.resize(img, (round(w * s), round(h * s)))
        yield t, base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()


async def main(args) -> int:
    states, latencies, predictions, alarms, errors = collections.Counter(), [], 0, 0, []
    async with websockets.connect(args.url, max_size=2**22) as ws:
        await ws.send(json.dumps({"type": "init"}))
        ack = json.loads(await ws.recv())
        print("init_ack:", ack)
        if ack.get("type") != "init_ack":
            return 1
        for t, b64 in frames(args.video, ack["target_fps"], args.seconds):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "frame", "ts": t * 1000, "image_b64": b64}))
            msg = json.loads(await ws.recv())
            latencies.append((time.perf_counter() - t0) * 1000)
            states[msg.get("state", msg.get("type"))] += 1
            predictions += bool(msg.get("predicted"))
            alarms += bool(msg.get("new_alarm"))
            if msg.get("type") == "error" or msg.get("state") == "error":
                errors.append(msg)
        await ws.send(json.dumps({"type": "close"}))
        summary = json.loads(await ws.recv())
    lat = sorted(latencies)
    report = {"frames": len(lat), "states": dict(states), "predictions": predictions, "alarms": alarms,
              "latency_ms_p50": round(lat[len(lat) // 2], 1), "latency_ms_p95": round(lat[int(len(lat) * 0.95)], 1),
              "errors": errors[:3], "server_summary": summary}
    print(json.dumps(report, indent=2))
    return 0 if predictions > 0 and not errors else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--seconds", type=float, default=90)
    sys.exit(asyncio.run(main(ap.parse_args())))
