"""
Per-session live feature extraction — the serving twin of the training pipeline.

* Frames are sampled by TIMESTAMP at params.serving.target_fps (same rate as training).
* Per-frame features: ddd_core.geometry.frame_features (same function as the pipeline).
* Window features: ddd_core.windows.window_features over the last `window_s` seconds, where
  window_s comes from the registered model's metadata (the window it was trained with).
MediaPipe is CPU-bound, so the caller runs process_frame in a worker thread.
"""

import base64
import binascii
import time
from collections import deque

import cv2
import numpy as np

from ddd_core.geometry import frame_features, landmarks_to_pixels
from ddd_core.windows import window_features


class FrameDecodeError(ValueError):
    pass


def decode_frame(image_b64: str) -> np.ndarray:
    if "," in image_b64[:100]:  # strip "data:image/jpeg;base64,"
        image_b64 = image_b64.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(image_b64, validate=True), dtype=np.uint8)
    except (binascii.Error, ValueError) as e:
        raise FrameDecodeError(f"invalid base64: {e}") from e
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FrameDecodeError("not a decodable image")
    return img


class LiveFeatureExtractor:
    def __init__(self, window_s: float, fps: float, max_side: int, feature_cfg: dict, face_mesh=None):
        self.window_s = float(window_s)
        self.fps = float(fps)
        self.max_side = int(max_side)
        self.cfg = feature_cfg
        self._mesh = face_mesh
        self.frames: deque = deque()  # (t, face, ear, mar, pitch, yaw, roll)
        self._next_t: float | None = None
        self.first_t: float | None = None
        self.last_face_t: float | None = None
        self._arrivals: deque = deque()  # timestamps of accepted frames (fps measurement)

    # ── sampling ───────────────────────────────────────────────────────────
    def should_sample(self, t: float) -> bool:
        """Accept at most target_fps frames per second, by timestamp (same rule as the extractor)."""
        if self._next_t is None:
            self._next_t = t
        if t + 1e-6 < self._next_t:
            return False
        step = 1.0 / self.fps
        while self._next_t <= t + 1e-6:
            self._next_t += step
        return True

    def _mesh_or_create(self):
        if self._mesh is None:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._mesh

    # ── per frame (blocking: run in a thread) ──────────────────────────────
    def process_frame(self, img: np.ndarray, t: float) -> dict:
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        res = self._mesh_or_create().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pts = landmarks_to_pixels(res.multi_face_landmarks[0].landmark, w, h) if res.multi_face_landmarks else None
        f = frame_features(pts, w, h)
        self.add_frame(t, f)
        return f

    def add_frame(self, t: float, f: dict) -> None:
        if self.first_t is None:
            self.first_t = t
        if f["face"]:
            self.last_face_t = t
        self.frames.append((t, f["face"], f["ear"], f["mar"], f["pitch"], f["yaw"], f["roll"]))
        self._arrivals.append(t)
        keep_from = t - self.window_s - 2.0
        while self.frames and self.frames[0][0] < keep_from:
            self.frames.popleft()
        while self._arrivals and self._arrivals[0] < t - 5.0:
            self._arrivals.popleft()

    # ── window ─────────────────────────────────────────────────────────────
    def window_ready(self, now: float) -> bool:
        return self.first_t is not None and now - self.first_t >= self.window_s

    def warmup_progress(self, now: float) -> float:
        if self.first_t is None:
            return 0.0
        return float(min(1.0, (now - self.first_t) / self.window_s))

    def compute_window(self, now: float) -> dict | None:
        # (now - L, now]: the same number of frames as the pipeline's [s, s + L) window.
        rows = [r for r in self.frames if now - self.window_s + 1e-6 < r[0] <= now]
        if not rows:
            return None
        cols = list(zip(*rows))
        return window_features(*cols, length_s=self.window_s, cfg=self.cfg, fps=self.fps)

    def measured_fps(self) -> float:
        if len(self._arrivals) < 2:
            return 0.0
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else 0.0

    def seconds_without_face(self, now: float) -> float:
        ref = self.last_face_t if self.last_face_t is not None else self.first_t
        return 0.0 if ref is None else max(0.0, now - ref)

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass


def now_s() -> float:
    return time.monotonic()
