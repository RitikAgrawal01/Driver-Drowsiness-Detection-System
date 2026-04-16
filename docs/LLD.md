# Low-Level Design (LLD)
## Driver Drowsiness Detection System — API Endpoint Specifications

**Version:** 1.0  
**Date:** April 2026  
**Authors:** RitikAgrawal01  

---

## Table of Contents
1. [Backend API — FastAPI (port 8000)](#1-backend-api--fastapi-port-8000)
2. [Model Server API — FastAPI (port 8001)](#2-model-server-api--fastapi-port-8001)
3. [WebSocket Protocol](#3-websocket-protocol)
4. [Internal Data Models (Pydantic Schemas)](#4-internal-data-models-pydantic-schemas)
5. [Module-Level Design](#5-module-level-design)
6. [Key Algorithms](#6-key-algorithms)
7. [Error Codes Reference](#7-error-codes-reference)

---

## 1. Backend API — FastAPI (port 8000)

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

### GET `/health`

**Purpose:** Docker healthcheck and load balancer readiness probe.

**Request:** No body, no parameters.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "service": "backend",
  "version": "1.0.0"
}
```

**Used by:** Docker Compose `healthcheck`, frontend status page.

---

### GET `/ready`

**Purpose:** Readiness check — returns `ok` only when the model server is reachable.

**Request:** No body.

**Response `200 OK` (model server reachable):**
```json
{
  "status": "ready",
  "model_server_reachable": true,
  "model_server_url": "http://model_server:8001"
}
```

**Response `503 Service Unavailable` (model server down):**
```json
{
  "status": "not_ready",
  "model_server_reachable": false,
  "detail": "Model server health check failed"
}
```

---

### POST `/session/start`

**Purpose:** Initialises a new drowsiness monitoring session. Returns a session ID used in subsequent WebSocket and predict calls.

**Request body:**
```json
{
  "driver_id": "string | null",
  "window_size": "integer | null (default: 30)"
}
```

**Response `200 OK`:**
```json
{
  "session_id": "uuid-string",
  "started_at": "2026-04-15T10:00:00Z",
  "window_size": 30,
  "status": "active"
}
```

**Response `422 Unprocessable Entity`:**
```json
{
  "detail": [
    {
      "loc": ["body", "window_size"],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

---

### POST `/session/stop`

**Purpose:** Terminates a monitoring session and returns a summary report.

**Request body:**
```json
{
  "session_id": "uuid-string"
}
```

**Response `200 OK`:**
```json
{
  "session_id": "uuid-string",
  "started_at": "2026-04-15T10:00:00Z",
  "ended_at": "2026-04-15T10:30:00Z",
  "duration_seconds": 1800,
  "total_frames_processed": 54000,
  "drowsy_alerts_triggered": 3,
  "average_ear": 0.29,
  "average_confidence": 0.82,
  "status": "completed"
}
```

**Response `404 Not Found`:**
```json
{
  "detail": "Session not found: <session_id>"
}
```

---

### GET `/session/{session_id}/status`

**Purpose:** Returns current real-time status of an active session.

**Path parameter:** `session_id` (string, UUID)

**Response `200 OK`:**
```json
{
  "session_id": "uuid-string",
  "status": "active",
  "frames_processed": 1200,
  "current_state": "alert",
  "current_confidence": 0.91,
  "current_ear": 0.30,
  "current_mar": 0.41,
  "current_perclos": 0.10,
  "alerts_triggered": 0,
  "drift_score": 0.03
}
```

**Response `404 Not Found`:**
```json
{
  "detail": "Session not found or already closed"
}
```

---

### POST `/predict`

**Purpose:** Accepts a precomputed feature vector (one sliding window) and returns a drowsiness prediction. Used for testing or batch inference — real-time sessions use the WebSocket endpoint instead.

**Request body:**
```json
{
  "session_id": "uuid-string | null",
  "features": {
    "ear_mean": 0.28,
    "ear_min": 0.19,
    "ear_std": 0.04,
    "perclos": 0.20,
    "mar_mean": 0.45,
    "mar_max": 0.72,
    "head_pitch_mean": -8.5,
    "head_yaw_mean": 3.2,
    "head_roll_mean": 1.1
  }
}
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `ear_mean` | float | Mean Eye Aspect Ratio over the window |
| `ear_min` | float | Minimum EAR (catches blinks) |
| `ear_std` | float | EAR standard deviation |
| `perclos` | float | Fraction of frames where EAR < threshold |
| `mar_mean` | float | Mean Mouth Aspect Ratio (yawn detection) |
| `mar_max` | float | Maximum MAR in window |
| `head_pitch_mean` | float | Mean head pitch in degrees (negative = nodding down) |
| `head_yaw_mean` | float | Mean head yaw in degrees |
| `head_roll_mean` | float | Mean head roll in degrees |

**Response `200 OK`:**
```json
{
  "state": "alert",
  "confidence": 0.93,
  "inference_latency_ms": 4.2,
  "model_name": "drowsiness_classifier",
  "model_version": "3",
  "alert_triggered": false
}
```

**Response `422 Unprocessable Entity`** (missing required feature):
```json
{
  "detail": [
    {
      "loc": ["body", "features", "ear_mean"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Response `503 Service Unavailable`** (model server unreachable):
```json
{
  "detail": "Model server unavailable. Please try again shortly."
}
```

---

### GET `/metrics`

**Purpose:** Prometheus scraping endpoint. Exposes all instrumented metrics.

**Response `200 OK`:** Plain text in Prometheus exposition format.

```
# HELP ddd_inference_latency_seconds End-to-end inference latency
# TYPE ddd_inference_latency_seconds histogram
ddd_inference_latency_seconds_bucket{le="0.05"} 1234
ddd_inference_latency_seconds_bucket{le="0.1"} 1456
ddd_inference_latency_seconds_bucket{le="0.2"} 1499
ddd_inference_latency_seconds_bucket{le="+Inf"} 1500
ddd_inference_latency_seconds_sum 72.3
ddd_inference_latency_seconds_count 1500

# HELP ddd_prediction_confidence Model prediction confidence score
# TYPE ddd_prediction_confidence histogram
...

# HELP ddd_feature_drift_score KL-divergence drift score per feature
# TYPE ddd_feature_drift_score gauge
ddd_feature_drift_score{feature="ear_mean"} 0.02
ddd_feature_drift_score{feature="perclos"} 0.11
...

# HELP ddd_requests_total Total API requests
# TYPE ddd_requests_total counter
ddd_requests_total{endpoint="/predict",status="200"} 1500
ddd_requests_total{endpoint="/predict",status="503"} 2

# HELP ddd_active_sessions Currently active monitoring sessions
# TYPE ddd_active_sessions gauge
ddd_active_sessions 1
```

---

## 2. Model Server API — FastAPI (port 8001)

Base URL: `http://localhost:8001`  
Interactive docs: `http://localhost:8001/docs`

> **Note:** The backend is the only caller of this API. The frontend never contacts the model server directly. This enforces loose coupling.

---

### GET `/health`

**Purpose:** Docker healthcheck.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "service": "model_server"
}
```

---

### GET `/ready`

**Purpose:** Readiness — returns `ok` only when the model has been loaded from the MLflow registry.

**Response `200 OK`:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "drowsiness_classifier",
  "model_version": "3",
  "model_stage": "Production"
}
```

**Response `503 Service Unavailable`:**
```json
{
  "status": "not_ready",
  "model_loaded": false,
  "detail": "Model not yet loaded from MLflow registry"
}
```

---

### POST `/predict`

**Purpose:** Runs inference on a precomputed feature vector. Called exclusively by the backend service.

**Request body:**
```json
{
  "ear_mean": 0.22,
  "ear_min": 0.14,
  "ear_std": 0.06,
  "perclos": 0.72,
  "mar_mean": 0.38,
  "mar_max": 0.51,
  "head_pitch_mean": -14.2,
  "head_yaw_mean": -2.1,
  "head_roll_mean": 0.8
}
```

**Response `200 OK`:**
```json
{
  "state": "drowsy",
  "confidence": 0.91,
  "inference_latency_ms": 3.7,
  "model_version": "3"
}
```

**Response `500 Internal Server Error`** (model inference failure):
```json
{
  "detail": "Inference failed: <error message>",
  "model_version": "3"
}
```

---

### GET `/model/info`

**Purpose:** Returns metadata about the currently loaded model.

**Response `200 OK`:**
```json
{
  "model_name": "drowsiness_classifier",
  "model_version": "3",
  "model_stage": "Production",
  "algorithm": "XGBoost",
  "mlflow_run_id": "abc123def456",
  "registered_at": "2026-04-14T08:30:00Z",
  "metrics": {
    "f1_score": 0.91,
    "accuracy": 0.92,
    "auc_roc": 0.96,
    "inference_latency_ms_p95": 4.1
  },
  "feature_names": [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean"
  ]
}
```

---

### GET `/metrics`

**Purpose:** Prometheus scraping endpoint for model server metrics.

**Response `200 OK`:** Prometheus exposition format.

```
# HELP ddd_model_inference_latency_seconds Model-only inference latency
# TYPE ddd_model_inference_latency_seconds histogram

# HELP ddd_model_prediction_confidence Prediction confidence from model
# TYPE ddd_model_prediction_confidence histogram

# HELP ddd_model_requests_total Total prediction requests
# TYPE ddd_model_requests_total counter
```

---

## 3. WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws`  
**Protocol:** WebSocket (RFC 6455)  
**Encoding:** JSON messages

### Connection Lifecycle

```
Client                          Server
  |                               |
  |── WS Upgrade (HTTP 101) ────► |
  |                               |
  |── {"type":"init", ...} ──────►|  Session setup
  |◄─ {"type":"init_ack", ...} ── |
  |                               |
  |── {"type":"frame", ...} ─────►|  Repeated per frame
  |◄─ {"type":"prediction", ...} ─|
  |                               |
  |── {"type":"close"} ──────────►|  Session teardown
  |◄─ {"type":"session_summary"} ─|
  |── WS Close ──────────────────►|
```

### Message: `init` (Client → Server)
```json
{
  "type": "init",
  "session_id": "uuid-string",
  "window_size": 30,
  "frame_rate": 30
}
```

### Message: `init_ack` (Server → Client)
```json
{
  "type": "init_ack",
  "session_id": "uuid-string",
  "status": "ready"
}
```

### Message: `frame` (Client → Server)
```json
{
  "type": "frame",
  "session_id": "uuid-string",
  "frame_id": 1042,
  "timestamp": "2026-04-15T10:05:22.341Z",
  "image_b64": "<base64-encoded JPEG string>"
}
```

### Message: `prediction` (Server → Client)
```json
{
  "type": "prediction",
  "session_id": "uuid-string",
  "frame_id": 1042,
  "state": "alert",
  "confidence": 0.94,
  "features": {
    "ear_mean": 0.30,
    "perclos": 0.08,
    "mar_mean": 0.40,
    "head_pitch_mean": -2.1
  },
  "alert_triggered": false,
  "inference_latency_ms": 18.4
}
```

### Message: `alert` (Server → Client, pushed when drowsy)
```json
{
  "type": "alert",
  "session_id": "uuid-string",
  "frame_id": 1190,
  "severity": "warning",
  "message": "Drowsiness detected! Please take a break.",
  "confidence": 0.89,
  "timestamp": "2026-04-15T10:06:00.112Z"
}
```

### Message: `close` (Client → Server)
```json
{
  "type": "close",
  "session_id": "uuid-string"
}
```

### Message: `session_summary` (Server → Client)
```json
{
  "type": "session_summary",
  "session_id": "uuid-string",
  "duration_seconds": 3600,
  "frames_processed": 108000,
  "alerts_triggered": 5,
  "average_confidence": 0.87,
  "average_ear": 0.28
}
```

---

## 4. Internal Data Models (Pydantic Schemas)

### `FeatureVector` (shared between backend and model server)
```python
class FeatureVector(BaseModel):
    ear_mean: float = Field(..., ge=0.0, le=1.0)
    ear_min: float = Field(..., ge=0.0, le=1.0)
    ear_std: float = Field(..., ge=0.0)
    perclos: float = Field(..., ge=0.0, le=1.0)
    mar_mean: float = Field(..., ge=0.0)
    mar_max: float = Field(..., ge=0.0)
    head_pitch_mean: float  # degrees, no clamp
    head_yaw_mean: float
    head_roll_mean: float
```

### `PredictionResponse`
```python
class PredictionResponse(BaseModel):
    state: Literal["alert", "drowsy"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_latency_ms: float
    model_version: str
    alert_triggered: bool
```

### `SessionStartRequest` / `SessionStopRequest`
```python
class SessionStartRequest(BaseModel):
    driver_id: Optional[str] = None
    window_size: int = Field(default=30, ge=10, le=300)

class SessionStopRequest(BaseModel):
    session_id: str
```

---

## 5. Module-Level Design

### Backend Module Structure
```
backend/
├── main.py                  # FastAPI app init, middleware, lifespan
├── config.py                # pydantic-settings: all env vars
├── routers/
│   ├── __init__.py
│   ├── predict.py           # POST /predict
│   ├── session.py           # POST /session/start, /stop, GET /status
│   └── stream.py            # WebSocket /ws
├── services/
│   ├── __init__.py
│   ├── feature_extractor.py # MediaPipe → EAR/MAR/PERCLOS/HeadPose
│   ├── model_client.py      # httpx async client → model server
│   ├── drift_detector.py    # KL-divergence vs baseline.json
│   └── alert_service.py     # Drowsiness threshold logic
├── metrics.py               # All prometheus_client instruments
├── logger.py                # structlog configuration
└── schemas.py               # All Pydantic models
```

### Model Server Module Structure
```
model_server/
├── main.py                  # FastAPI app, model load on startup
├── config.py                # Env vars: MLflow URI, model name/stage
├── model_loader.py          # MLflow registry → load model to memory
├── predict.py               # Inference logic, latency measurement
├── metrics.py               # Prometheus instruments
├── train_xgboost.py         # Training script (Phase 4)
├── train_svm.py             # Training script (Phase 4)
├── evaluate.py              # Evaluation + MLflow registry promotion
└── schemas.py               # FeatureVector, PredictionResponse
```

---

## 6. Key Algorithms

### 6.1 Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

Where p1..p6 are the 6 eye landmark points from MediaPipe.
Landmark indices (left eye):  33, 160, 158, 133, 153, 144
Landmark indices (right eye): 362, 385, 387, 263, 373, 380
```
EAR < 0.25 → eye considered closed for that frame.

### 6.2 PERCLOS
```
PERCLOS = (frames where EAR < EAR_THRESHOLD) / total_frames_in_window
```
PERCLOS > 0.8 over a 30-frame window → strong drowsiness indicator.

### 6.3 Mouth Aspect Ratio (MAR)
```
MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 × ||p1-p5||)

Where p1..p8 are mouth landmark points.
```
MAR > 0.6 → yawn detected.

### 6.4 Head Pose Estimation
Uses OpenCV `solvePnP` with a set of 3D model reference points (nose tip, chin, eye corners, mouth corners) matched to corresponding 2D MediaPipe landmarks. Returns rotation vector → converted to Euler angles (pitch, yaw, roll) via Rodrigues decomposition.

### 6.5 Drift Detection (KL Divergence)
For each feature f, compare live distribution (rolling 5-minute window) vs baseline distribution:
```
drift_score(f) = KL_divergence(P_live(f) || P_baseline(f))
```
Where distributions are approximated as Gaussians using mean and variance.  
Overall drift score = max(drift_score(f) for all features f).

---

## 7. Error Codes Reference

| HTTP Code | Scenario | Example |
|---|---|---|
| 200 | Success | Prediction returned |
| 422 | Validation error | Missing feature field |
| 404 | Resource not found | Unknown session_id |
| 503 | Upstream unavailable | Model server down |
| 500 | Internal error | Unexpected exception |

All error responses follow the format:
```json
{
  "detail": "Human-readable error message",
  "error_code": "OPTIONAL_CODE_STRING"
}
```
