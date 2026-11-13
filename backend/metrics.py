"""
backend/metrics.py — Prometheus instruments (exposed at /metrics).

Counting rules (v1 mixed units, so its error ratio could exceed 100 %):
  * frames are counted per frame, predictions per model call, errors per failed model call;
  * error ratio = ddd_prediction_errors_total / ddd_predictions_total (same unit).
"""

from prometheus_client import Counter, Gauge, Histogram

FRAMES_RECEIVED = Counter("ddd_frames_received_total", "Frames received over WebSocket")
FRAMES_PROCESSED = Counter("ddd_frames_processed_total", "Frames run through MediaPipe")
FRAMES_SKIPPED = Counter("ddd_frames_skipped_total", "Frames not processed", ["reason"])

PREDICTIONS = Counter("ddd_predictions_total", "Model calls (one per new window)", ["source"])
PREDICTION_ERRORS = Counter("ddd_prediction_errors_total", "Failed model calls", ["source", "error_type"])
PREDICTION_SCORE = Histogram(
    "ddd_prediction_score",
    "Model score (higher = more drowsy)",
    buckets=[-2, -1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2],
)

FRAME_LATENCY = Histogram(
    "ddd_frame_latency_seconds",
    "Frame received -> status sent (end to end)",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)
FEATURE_EXTRACTION_LATENCY = Histogram(
    "ddd_feature_extraction_latency_seconds",
    "MediaPipe + geometry per frame",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
)
MODEL_CALL_LATENCY = Histogram(
    "ddd_model_call_latency_seconds",
    "Backend -> model server round trip",
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)
# Kept for the existing dashboards / SLA rule: end-to-end latency of a prediction-bearing frame.
INFERENCE_LATENCY = Histogram(
    "ddd_inference_latency_seconds",
    "Frame -> prediction (frames that produced one)",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

ACTIVE_SESSIONS = Gauge("ddd_active_sessions", "Active monitoring sessions")
DROWSY_ALARMS = Counter("ddd_drowsy_alerts_total", "Alarm switched ON (edge-triggered)")
STATE_TRANSITIONS = Counter("ddd_state_transitions_total", "Displayed state changes", ["to_state"])

FEATURE_PSI = Gauge("ddd_feature_drift_psi", "PSI per feature vs training baseline", ["feature"])
OVERALL_DRIFT = Gauge("ddd_overall_drift_score", "Max PSI across features (-1 = not enough data)")
DRIFT_WINDOWS = Gauge("ddd_drift_buffer_windows", "Live windows in the drift buffer")
DRIFT_BASELINE_LOADED = Gauge("ddd_drift_baseline_loaded", "1 if the champion's drift baseline is loaded")

MODEL_SERVER_REACHABLE = Gauge("ddd_model_server_reachable", "1 if the model server is ready")
SERVED_MODEL_VERSION = Gauge("ddd_served_model_version", "Registered model version the backend is using")
