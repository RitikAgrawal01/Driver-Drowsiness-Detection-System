"""
backend/metrics.py
───────────────────
All Prometheus metric instruments for the backend service.
Defined once here and imported wherever needed.
Exposed at GET /metrics endpoint.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Inference latency ─────────────────────────────────────────────────────────
# End-to-end: from frame received to prediction returned to client
INFERENCE_LATENCY = Histogram(
    "ddd_inference_latency_seconds",
    "End-to-end inference latency (frame → prediction)",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

# ── Prediction confidence distribution ───────────────────────────────────────
PREDICTION_CONFIDENCE = Histogram(
    "ddd_prediction_confidence",
    "Model prediction confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Request counters ──────────────────────────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "ddd_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

REQUEST_ERRORS_TOTAL = Counter(
    "ddd_request_errors_total",
    "Total failed requests",
    ["endpoint", "error_type"],
)

# ── Session metrics ───────────────────────────────────────────────────────────
ACTIVE_SESSIONS = Gauge(
    "ddd_active_sessions",
    "Currently active monitoring sessions",
)

FRAMES_PROCESSED_TOTAL = Counter(
    "ddd_frames_processed_total",
    "Total webcam frames processed",
)

DROWSY_ALERTS_TOTAL = Counter(
    "ddd_drowsy_alerts_total",
    "Total drowsiness alerts triggered",
)

# ── Data drift ────────────────────────────────────────────────────────────────
FEATURE_DRIFT_SCORE = Gauge(
    "ddd_feature_drift_score",
    "KL-divergence drift score per feature vs training baseline",
    ["feature"],
)

OVERALL_DRIFT_SCORE = Gauge(
    "ddd_overall_drift_score",
    "Maximum drift score across all features",
)

# ── Model server health ───────────────────────────────────────────────────────
MODEL_SERVER_REACHABLE = Gauge(
    "ddd_model_server_reachable",
    "1 if model server is reachable, 0 otherwise",
)

# ── Pipeline throughput ───────────────────────────────────────────────────────
FEATURE_EXTRACTION_LATENCY = Histogram(
    "ddd_feature_extraction_latency_seconds",
    "Time to extract features from one frame with MediaPipe",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
)
