# High-Level Design (HLD)
## Driver Drowsiness Detection System with End-to-End MLOps Pipeline

**Version:** 1.0  
**Date:** April 2026  
**Authors:** RitikAgrawal01  

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Design Goals & Constraints](#2-design-goals--constraints)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Descriptions](#4-component-descriptions)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Rationale](#6-design-decisions--rationale)
7. [MLOps Lifecycle Coverage](#7-mlops-lifecycle-coverage)
8. [Security Considerations](#8-security-considerations)
9. [Scalability & Performance](#9-scalability--performance)

---

## 1. System Overview

The Driver Drowsiness Detection System (DDD) is a real-time AI application that monitors a driver's facial behaviour via a webcam feed and triggers an alert when signs of drowsiness are detected. The system is backed by a fully automated MLOps infrastructure covering data ingestion, model training, experiment tracking, deployment, and monitoring.

**Core capabilities:**
- Real-time drowsiness detection at ≥ 15 FPS with inference latency < 200ms (SLA)
- Geometric feature extraction: Eye Aspect Ratio (EAR), PERCLOS, Mouth Aspect Ratio (MAR), Head Pose (pitch, yaw, roll) using MediaPipe's 468-point facial landmark model
- Classification using XGBoost (primary) and SVM (comparison) models, both tracked in MLflow
- Fully Dockerised microservice architecture with loose coupling between frontend and backend via REST APIs
- Automated retraining when model performance degrades beyond a defined threshold

---

## 2. Design Goals & Constraints

### Goals
| Goal | Metric |
|---|---|
| Real-time inference | Latency < 200ms (P95) |
| High accuracy | F1-score > 0.85 on test set |
| Non-technical usability | Intuitive UI with user manual |
| Full MLOps coverage | Airflow, DVC, MLflow, Prometheus, Grafana, Docker |
| Reproducibility | Every experiment reproducible via Git commit + MLflow run ID |
| Loose coupling | Frontend ↔ Backend ↔ Model Server connected only via configurable REST APIs |

### Constraints
- **No cloud**: All components run on local/on-premise hardware only
- **No commercial monitoring**: Only Prometheus + Grafana (open-source)
- **Model optimisation**: Models must be optimised for local CPU inference (no GPU assumed)

---

## 3. Architecture Overview

The system is divided into four horizontal layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: FRONTEND  (React + Vite, port 3000)                   │
│  Live Monitor | Pipeline Console | Monitoring | Model Registry   │
└────────────────────────┬────────────────────────────────────────┘
                         │ REST + WebSocket
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 2: BACKEND  (FastAPI, port 8000)                         │
│  WebSocket Handler | Feature Extractor | Drift Detector | API   │
└────────────────────────┬────────────────────────────────────────┘
                         │ REST /predict
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 3: ML LAYER                                              │
│  Model Server (port 8001) ←→ MLflow Registry (port 5000)        │
│  XGBoost (primary) | SVM (comparison)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ orchestration / versioning
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 4: MLOPS                                                 │
│  Airflow (8080) | DVC Pipeline | Prometheus (9090) | Grafana (3001) │
└─────────────────────────────────────────────────────────────────┘
```

All services run as separate Docker containers managed by `docker-compose.yml` on a shared `ddd_network` bridge network.

---

## 4. Component Descriptions

### 4.1 Frontend (React + Vite)
A single-page application with four pages, served by Nginx in production:

| Page | Description |
|---|---|
| Live Monitor | Renders webcam feed, real-time EAR/MAR gauges, drowsiness alert banner, audio alert, session controls |
| Pipeline Console | Displays Airflow DAG status, pipeline run history, task error/success log |
| Monitoring Dashboard | Embeds Grafana panels showing inference latency, drift score, error rate |
| Model Registry | Lists MLflow experiments, compares metrics, shows current production model |

The frontend is **strictly decoupled** from the backend. All communication is via configurable REST API calls (URL set via `VITE_BACKEND_URL` env var) and a WebSocket connection (`VITE_WS_URL`) for real-time frame streaming. The frontend has zero direct knowledge of the ML stack.

### 4.2 Backend (FastAPI, port 8000)
The main orchestration service. Responsibilities:
- **WebSocket handler** (`/ws`): Receives base64-encoded webcam frames from the browser, manages session state
- **Feature extractor**: Runs MediaPipe FaceMesh on each frame to extract 468 3D facial landmarks. Computes EAR, MAR, PERCLOS, and head pose (pitch/yaw/roll via solvePnP) over a configurable sliding window (default: 30 frames = 1 second at 30 FPS)
- **Model client**: Calls the model server's `/predict` endpoint via HTTP with the computed feature vector
- **Drift detector**: Continuously compares incoming feature distributions against the training-time `baseline.json` using KL-divergence. Emits a `ddd_feature_drift_score` Prometheus gauge
- **Prometheus instrumentation**: Exposes `/metrics` endpoint with inference latency histogram, prediction confidence histogram, request counters, error counters, and active session gauge
- **Logging**: Structured logging via `structlog` to rotating files in `logs/backend/`
- **Exception handling**: All endpoints wrapped with try/except; appropriate HTTP 4xx/5xx responses with logged stack traces

### 4.3 Model Server (FastAPI, port 8001)
A lightweight inference-only service:
- On startup, loads the registered `Production` model from the MLflow Model Registry
- Exposes `/predict` (POST): accepts a feature vector JSON, returns `{state: "alert"|"drowsy", confidence: float}`
- Exposes `/health` and `/ready` endpoints for Docker healthchecks and orchestration
- Exposes `/metrics` for Prometheus scraping (inference latency, prediction confidence)
- The active model (XGBoost vs SVM) is set via `MLFLOW_MODEL_NAME` env var — no code change needed to switch models

### 4.4 MLflow (port 5000)
Tracks all training experiments:
- **Experiment tracking**: Every training run logs parameters (hyperparameters, window size, dataset hash), metrics (F1, accuracy, precision, recall, AUC-ROC, inference latency), and artifacts (model `.pkl`, confusion matrix PNG, feature importance PNG)
- **Model Registry**: Best model promoted to `Production` stage after evaluation. Previous version moved to `Archived`
- **Autolog + manual logging**: MLflow autolog captures XGBoost/sklearn basics; additional manual logging captures dataset DVC commit hash and Git commit SHA for full reproducibility
- **Backend store**: SQLite (`mlflow.db`) inside the container; artifact store on a named Docker volume

### 4.5 Apache Airflow (port 8080)
Orchestrates the data engineering pipeline:
- **`dag_data_pipeline`**: Triggered manually or on a schedule. Tasks: (1) extract frames from raw video using OpenCV at configurable FPS, (2) run MediaPipe on each frame to extract landmarks and save as CSV, (3) run PySpark job for EAR/MAR/PERCLOS/head-pose feature engineering over sliding windows, (4) compute baseline statistics and save `baseline.json`, (5) `dvc add` + `dvc push` versioned outputs
- **`dag_retrain`**: Triggered by Prometheus alert webhook when `ddd_feature_drift_score > 0.15` or F1 drops > 2%. Pulls latest DVC data, retrains, evaluates, and auto-promotes if new model exceeds production F1 by > 2%
- Pipeline throughput (frames/sec, seconds per stage) is logged as Airflow XComs and emitted to Prometheus

### 4.6 DVC Pipeline
Version-controls all data and model artifacts:
- **`dvc.yaml`** defines 7 reproducible stages: `extract_frames → extract_landmarks → feature_engineering → split_data → train_xgboost → train_svm → evaluate`
- **`params.yaml`** centralises all tunable parameters
- `dvc dag` produces a visual DAG of the pipeline
- Every experiment is reproducible via `git checkout <hash> && dvc repro`
- Git LFS tracks binary files (`.pkl`, `.mp4`, `.h5`)

### 4.7 Prometheus (port 9090)
Scrapes metrics from backend and model server every 10 seconds. Monitors:
- `ddd_inference_latency_seconds` (histogram)
- `ddd_prediction_confidence` (histogram)
- `ddd_feature_drift_score` (gauge, per feature)
- `ddd_requests_total` / `ddd_request_errors_total` (counters)
- `ddd_active_sessions` (gauge)

Alert rules fire when: error rate > 5%, P95 latency > 200ms, drift score > 0.15, model server down.

### 4.8 Grafana (port 3001)
Visualises Prometheus data in near-real-time (NRT):
- Provisioned automatically via `monitoring/grafana/provisioning/` — no manual setup
- Dashboard panels: inference latency time series, prediction confidence distribution, per-feature drift scores, error rate %, active sessions counter
- Embedded in the frontend's Monitoring Dashboard page via iframe

---

## 5. Data Flow

### 5.1 Real-Time Inference Flow
```
Browser webcam
    │ base64 frame (WebSocket /ws)
    ▼
Backend: MediaPipe → 468 landmarks
    │
    ▼
Backend: Compute EAR, MAR, PERCLOS, head pose (sliding window 30 frames)
    │
    ├──► Drift Detector (async, compare vs baseline.json)
    │
    ▼ REST POST /predict
Model Server: XGBoost inference
    │
    ▼ {state, confidence}
Backend: threshold check → alert if drowsy + confidence > 0.7
    │
    ▼ WebSocket push
Browser: update UI alert banner + audio
```

### 5.2 Training Pipeline Flow
```
data/raw/ (video files, DVC-tracked)
    │ Airflow Task 1: OpenCV frame extraction
    ▼
data/frames/ (PNG frames, DVC-tracked)
    │ Airflow Task 2: MediaPipe landmark extraction
    ▼
data/landmarks/ (CSVs, DVC-tracked)
    │ Airflow Task 3: PySpark feature engineering
    ▼
data/features/features.csv + baseline.json (DVC-tracked)
    │ split_data stage
    ▼
data/processed/train.csv + test.csv
    │ train_xgboost + train_svm stages (MLflow tracking)
    ▼
models/*.pkl (DVC-tracked) + MLflow artifacts
    │ evaluate stage
    ▼
reports/evaluation_report.json
    │ manual: promote best model in MLflow Registry
    ▼
Model Server: loads Production model on startup
```

---

## 6. Design Decisions & Rationale

### 6.1 Why MediaPipe FaceMesh?
MediaPipe provides 468 3D facial landmarks at real-time speed (>30 FPS on CPU) without requiring a GPU. It is the industry standard for lightweight facial geometry extraction and eliminates the need for a custom face detector. The 468-landmark mesh gives sufficient geometric information to compute all required drowsiness features (EAR, MAR, head pose).

### 6.2 Why XGBoost as the primary model?
XGBoost is a gradient-boosted tree ensemble that offers state-of-the-art tabular performance, fast CPU inference (< 5ms per prediction), interpretable feature importance, and no GPU requirement. Given that our features are a compact vector (7–10 floats per window), XGBoost is well-suited. SVM is included as a comparison model to satisfy the guideline requirement of experimenting with multiple algorithms.

### 6.3 Why FastAPI for both backend and model server?
FastAPI provides automatic OpenAPI documentation (`/docs`), async support (critical for WebSocket), Pydantic validation for all I/O, and a minimal overhead Uvicorn ASGI server. The strict separation into two independent FastAPI services (backend on 8000, model server on 8001) enforces the guideline's **loose coupling** requirement: the frontend never calls the model server directly, and the model server has no knowledge of the frontend.

### 6.4 Why Airflow + PySpark for the data pipeline?
The guideline explicitly requires one of Airflow, Ray, Spark, or a custom pipeline. Airflow provides DAG-based orchestration with a built-in UI for pipeline management and error tracking. PySpark handles the sliding-window feature engineering step at scale, and its DataFrame API produces clean, version-controlled feature CSVs.

### 6.5 Why DVC for versioning?
DVC is the guideline's specified tool for data and model versioning (Source Control & CI [2 pts]). It integrates with Git to version datasets and models without storing binary blobs in Git itself, uses a content-addressed cache, and produces a reproducible DAG (`dvc dag`) that serves as the CI pipeline.

### 6.6 Why Docker Compose with separate containers?
The guideline requires Dockerisation with separate containers for frontend, backend, and model server, connected via REST APIs. Docker Compose manages the multi-service lifecycle, shared networking (`ddd_network`), named volumes for persistence, and environment variable injection — all without cloud infrastructure.

---

## 7. MLOps Lifecycle Coverage

| Lifecycle Stage | Implementation |
|---|---|
| Problem definition | EAR < 0.25 for > 0.8 PERCLOS over 30 frames → drowsy |
| Data collection | UTA-RLDD dataset + custom recordings |
| Data validation | Schema + missing value checks as Airflow task |
| EDA & baseline | Feature distributions computed in feature_engineering.py → baseline.json |
| Feature engineering | PySpark sliding-window EAR, MAR, PERCLOS, head pose |
| Model training | XGBoost + SVM with MLflow autolog + manual tracking |
| Model evaluation | F1, accuracy, AUC-ROC, inference latency; comparison in MLflow UI |
| Model registry | MLflow registry: Staging → Production promotion |
| Deployment | FastAPI model server loading from registry; Docker Compose |
| Monitoring | Prometheus + Grafana: latency, confidence, drift, error rate |
| Retraining | Airflow `dag_retrain` triggered by Prometheus alert or F1 decay |
| CI/CD | GitHub Actions: lint → tests → docker build; DVC repro |

---

## 8. Security Considerations

- All sensitive configuration (credentials, tokens) stored in `.env` — never committed to Git (`.gitignore`)
- Docker containers run as non-root user (`appuser`, UID 1000)
- Grafana admin credentials set via environment variables
- Data at rest: raw video and feature CSVs stored on local filesystem; no external transmission
- Model artifacts stored in MLflow's local artifact store (Docker named volume)

---

## 9. Scalability & Performance

- **Sliding window size** is configurable via `SLIDING_WINDOW_SIZE` env var — trade-off between responsiveness and accuracy
- **XGBoost model** is optimised for CPU inference: tree depth tuning, subsampling to prevent overfitting
- **Docker Compose** can be extended to Docker Swarm for multi-node deployment if hardware allows
- **Prometheus retention** is configurable; current default is 15 days of time-series data
- **Airflow LocalExecutor** is used for single-machine deployment; can be upgraded to CeleryExecutor for distributed execution
