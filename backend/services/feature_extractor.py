"""
backend/services/feature_extractor.py
───────────────────────────────────────
Real-time feature extraction service.

Receives raw webcam frames (base64 JPEG), runs MediaPipe FaceMesh,
computes EAR, MAR, PERCLOS, head pose over a sliding window of frames,
and returns a FeatureVector ready for inference.

This is the most compute-intensive part of the backend.
MediaPipe runs in VIDEO mode (not STATIC_IMAGE_MODE) for the live stream,
which uses temporal tracking for better landmark stability.
"""

import base64
import math
import time
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from backend.logger import get_logger
from backend.metrics import FEATURE_EXTRACTION_LATENCY

logger = get_logger("feature_extractor")

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

# Head pose reference points (3D model coords in mm, approximate)
HEAD_3D_POINTS = np.array([
    [0.0,    0.0,    0.0  ],   # nose tip        (lm 1)
    [0.0,   -63.6, -12.5 ],   # chin            (lm 152)
    [-43.3,  32.7,  26.0 ],   # left eye corner (lm 226)
    [43.3,   32.7,  26.0 ],   # right eye corner(lm 446)
    [-28.9, -28.9,  24.1 ],   # left mouth      (lm 57)
    [28.9,  -28.9,  24.1 ],   # right mouth     (lm 287)
], dtype=np.float64)

HEAD_LM_INDICES = [1, 152, 226, 446, 57, 287]


class PerFrameFeatures:
    """Stores computed features for one frame."""
    __slots__ = ["ear", "mar", "eye_closed", "pitch", "yaw", "roll",
                 "timestamp", "face_detected"]

    def __init__(self, ear=0.0, mar=0.0, eye_closed=False,
                 pitch=0.0, yaw=0.0, roll=0.0,
                 timestamp=0.0, face_detected=True):
        self.ear = ear
        self.mar = mar
        self.eye_closed = eye_closed
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.timestamp = timestamp
        self.face_detected = face_detected


class FeatureExtractor:
    """
    Stateful feature extractor for a single monitoring session.
    Maintains a sliding window of per-frame features.
    One instance per WebSocket session.
    """

    def __init__(self, window_size: int = 30,
                 ear_threshold: float = 0.25,
                 mar_threshold: float = 0.60):
        self.window_size = window_size
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        
        self.frame_counter = 0

        # Sliding window (deque auto-discards old frames)
        self._window: deque = deque(maxlen=window_size)

        # MediaPipe FaceMesh — VIDEO mode for live streaming
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,       # VIDEO mode: uses tracking
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Camera intrinsics — estimated for webcam
        # Actual values depend on webcam; these are reasonable defaults
        self._cam_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1))

        logger.info(f"FeatureExtractor initialised (window={window_size})")

    def _get_cam_matrix(self, w: int, h: int) -> np.ndarray:
        """Lazy-initialise camera matrix from frame dimensions."""
        if self._cam_matrix is None:
            focal = w  # rough estimate: focal length ≈ image width
            self._cam_matrix = np.array([
                [focal, 0,     w / 2],
                [0,     focal, h / 2],
                [0,     0,     1    ],
            ], dtype=np.float64)
        return self._cam_matrix

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _compute_ear(self, lms, indices: list, w: int, h: int) -> float:
        """Eye Aspect Ratio = (V1 + V2) / (2 * H)."""
        p = [(lms[i].x * w, lms[i].y * h) for i in indices]
        v1 = self._dist(p[1], p[5])
        v2 = self._dist(p[2], p[4])
        hd = self._dist(p[0], p[3])
        return (v1 + v2) / (2.0 * hd) if hd > 1e-6 else 0.0

    def _compute_mar(self, lms, indices: list, w: int, h: int) -> float:
        p = [(lms[i].x * w, lms[i].y * h) for i in indices]
        horizontal = self._dist(p[0], p[1])
        vertical = self._dist(p[4], p[5]) 
        return vertical / horizontal if horizontal > 1e-6 else 0.0

    def _compute_head_pose(self, lms, w: int, h: int) -> tuple:
        """
        Estimate head pose (pitch, yaw, roll) using solvePnP.
        Returns (pitch_deg, yaw_deg, roll_deg).
        """
        try:
            img_pts = np.array([
                [lms[i].x * w, lms[i].y * h]
                for i in HEAD_LM_INDICES
            ], dtype=np.float64)

            cam_matrix = self._get_cam_matrix(w, h)
            success, rot_vec, trans_vec = cv2.solvePnP(
                HEAD_3D_POINTS, img_pts,
                cam_matrix, self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return 0.0, 0.0, 0.0

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            # Decompose rotation matrix to Euler angles
            sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                pitch = math.degrees(math.atan2( rot_mat[2, 1], rot_mat[2, 2]))
                yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
                roll  = math.degrees(math.atan2( rot_mat[1, 0], rot_mat[0, 0]))
            else:
                pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
                yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
                roll  = 0.0

            return round(pitch, 3), round(yaw, 3), round(roll, 3)

        except Exception as e:
            logger.debug(f"Head pose estimation failed: {e}")
            return 0.0, 0.0, 0.0

    # ── Main processing entry point ───────────────────────────────────────────

    def process_frame(self, frame_b64: str) -> Optional[dict]:
        """
        Process one base64-encoded JPEG frame.

        1. Decode JPEG → numpy BGR image
        2. Run MediaPipe FaceMesh
        3. Compute EAR, MAR, head pose
        4. Push to sliding window
        5. If window is full → compute and return FeatureVector dict
           If window not full yet → return None

        Args:
            frame_b64: base64-encoded JPEG string from webcam

        Returns:
            dict with feature vector keys, or None if window not full
        """
        t_start = time.perf_counter()

        # 1. Increment the counter
        self.frame_counter += 1
        
        # 2. Check if we should skip this frame (Targeting ~2 FPS from a 15 FPS stream)
        if self.frame_counter < 7:
            # If window is already full, return existing aggregation so UI stays live
            if len(self._window) >= self.window_size:
                return self._aggregate_window()
            return None
        
        # 3. If we reached 7, reset and proceed with heavy extraction
        self.frame_counter = 0

        # ── Decode frame ──────────────────────────────────────────────────────
        try:
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                logger.warning("Failed to decode frame — skipping")
                return None
        except Exception as e:
            logger.warning(f"Frame decode error: {e}")
            return None

        # Force dimensions to match the training baseline exactly
        w, h = 640, 480
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── MediaPipe ─────────────────────────────────────────────────────────
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            # No face detected — push neutral values to avoid breaking window
            pff = PerFrameFeatures(face_detected=False,
                                   timestamp=time.time())
            self._window.append(pff)

            elapsed = (time.perf_counter() - t_start)
            FEATURE_EXTRACTION_LATENCY.observe(elapsed)
            return None  # don't infer without face

        lms = results.multi_face_landmarks[0].landmark

        # ── Per-frame features ────────────────────────────────────────────────
        right_ear = self._compute_ear(lms, RIGHT_EYE, w, h)
        left_ear  = self._compute_ear(lms, LEFT_EYE, w, h)
        ear = (right_ear + left_ear) / 2.0

        mar = self._compute_mar(lms, MOUTH, w, h)
        pitch, yaw, roll = self._compute_head_pose(lms, w, h)
        eye_closed = ear < self.ear_threshold

        pff = PerFrameFeatures(
            ear=ear, mar=mar,
            eye_closed=eye_closed,
            pitch=pitch, yaw=yaw, roll=roll,
            timestamp=time.time(),
            face_detected=True,
        )
        self._window.append(pff)

        elapsed = time.perf_counter() - t_start
        FEATURE_EXTRACTION_LATENCY.observe(elapsed)

        # ── Window aggregation ────────────────────────────────────────────────
        if len(self._window) < self.window_size:
            return None  # wait for full window

        return self._aggregate_window()

    def _aggregate_window(self) -> dict:
        """Aggregate the current sliding window into a feature vector."""
        frames = list(self._window)
        valid = [f for f in frames if f.face_detected]

        if not valid:
            return None

        ears   = [f.ear   for f in valid]
        mars   = [f.mar   for f in valid]
        pitches = [f.pitch for f in valid]
        yaws   = [f.yaw   for f in valid]
        rolls  = [f.roll  for f in valid]
        closed = [f.eye_closed for f in frames]  # all frames for PERCLOS

        return {
            "ear_mean": round(float(np.mean(ears)),  6),
            "ear_min":  round(float(np.min(ears)),   6),
            "ear_std":  round(float(np.std(ears)),   6),
            "perclos":  round(float(np.mean(closed)), 6),
            "mar_mean": round(float(np.mean(mars)),  6),
            "mar_max":  round(float(np.max(mars)),   6),
            "head_pitch_mean": round(float(np.mean(pitches)), 4),
            "head_yaw_mean":   round(float(np.mean(yaws)),    4),
            "head_roll_mean":  round(float(np.mean(rolls)),   4),
        }

    def get_current_ear(self) -> Optional[float]:
        """Return the most recent EAR value (for status endpoint)."""
        valid = [f for f in self._window if f.face_detected]
        return valid[-1].ear if valid else None

    def get_current_mar(self) -> Optional[float]:
        valid = [f for f in self._window if f.face_detected]
        return valid[-1].mar if valid else None

    def get_current_perclos(self) -> Optional[float]:
        if not self._window:
            return None
        closed = [f.eye_closed for f in self._window]
        return round(float(np.mean(closed)), 4)

    def close(self):
        """Release MediaPipe resources."""
        try:
            self._face_mesh.close()
        except Exception:
            pass
