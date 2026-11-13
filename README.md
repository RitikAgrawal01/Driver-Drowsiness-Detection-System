# Driver Drowsiness Detection System

Real-time drowsiness detection from a webcam: MediaPipe facial landmarks → eye/mouth/head-pose
and blink features over time windows → a classifier served behind a WebSocket API. Built
around a full MLOps loop: DVC pipeline, MLflow tracking and registry, Airflow orchestration,
Prometheus/Grafana/Alertmanager monitoring, Docker Compose, and GitHub Actions CI.

> **Two versions.** `v1` (tag [`v1-course-submission`](../../tree/v1-course-submission)) is the
> course project as submitted. `v2` (this branch) is a rebuild after a code audit: it fixes data
> leakage, two sources of truth, fail-open serving and a CI that could not fail. See
> [Project history](#project-history-v1--v2).

## Results (v2, honest evaluation)

24 participants from UTA-RLDD (Folds 1–2), 48 videos (alert = KSS 0, drowsy = KSS 10), 10 fps,
evaluated with **nested leave-one-participant-out**: every number below is measured on a
person the model never saw. Fold 4 is **sealed** as a final test set; every stage refuses it.
Full tables: [`reports/RESULTS.md`](reports/RESULTS.md).

| Model (main features) | ROC-AUC per held-out person | Pooled F1 | Videos correct |
|---|---|---|---|
| **PERCLOS rule (baseline)** | **0.86 ± 0.24** | 0.74 | 35 / 48 |
| XGBoost | 0.82 ± 0.21 | 0.71 | 33 / 48 |
| Logistic regression | 0.74 ± 0.25 | 0.64 | 26 / 48 |
| SVM (RBF) | 0.64 ± 0.33 | 0.67 | 29 / 48 |

- **No ML model reliably beats the rule:** paired by person, XGBoost without head pose
  (0.83 ± 0.18) is better for 8 of 24 people (Wilcoxon p = 0.12). The gate therefore deploys the
  rule as `@champion` (policy), comparing against the champion re-scored on the same data.
- **Leakage, measured:** with the same features and v1's settings, XGBoost scores F1 0.99 /
  AUC 1.00 on v1's random window split but 0.60 / 0.61 on unseen people.
- **Head pose is a camera-setup shortcut:** several participants changed phone orientation or
  resolution between their two videos; removing pose features helps logistic regression
  (0.74 → 0.77) and the SVM (0.64 → 0.74).
- **Per-driver centring** (exploratory, [`reports/EXPLORATORY_CALIBRATION.md`](reports/EXPLORATORY_CALIBRATION.md)):
  subtracting each driver's own median does not change per-person ranking, but lifts the rule's
  pooled specificity from 0.43 to 0.81, i.e. far fewer false alarms with one threshold.

## Architecture

```
 Browser (React) ── WebSocket: 10 fps timestamped JPEGs, one frame in flight ──► Backend (FastAPI)
                                                                               │ MediaPipe (worker thread)
                                                                               │ ddd_core features (same code as training)
                                                                               │ every 2 s: 12 window features
                                                                               ▼
 MLflow registry ── models:/drowsiness_classifier_v2@champion ──► Model server (pyfunc, threshold inside)

 Prometheus ── scrapes backend / model server ──► alert rules (generated from params.yaml)
      └──► Alertmanager ── action=retrain ──► webhook ──► Airflow ddd_retrain_pipeline ──► dvc repro
                         └─ action=page ────► webhook /notify (log; swap for Slack/PagerDuty)
```

**One source of truth.** Every behavioural setting (fps, windows, thresholds, alarm timing,
drift, SLOs, gate rules) lives in [`params.yaml`](params.yaml): DVC stages, Airflow (which only
runs `dvc repro`), the backend and the model server all read it, and the alert rules and
dashboard thresholds are generated from it. `.env` holds infrastructure only. CI fails if
`dvc.lock` was produced with different params.

## Training pipeline (DVC, 7 stages)

| Stage | What it does |
|---|---|
| `extract_landmarks` | video → 10 fps by timestamp, rotation metadata applied, MediaPipe Face Mesh (video mode) |
| `frame_features` | per frame: EAR, inner-lip MAR, head pose (`ddd_core.geometry`, shared with serving) |
| `window_features` | 15/30/60 s windows, 2 s stride: 9 v1 features (fixed) + blink rate, blink duration, long closures |
| `train_evaluate` | nested LOSO: PERCLOS rule, logistic regression, SVM, XGBoost × 3 feature sets; window length, hyperparameters, trees and threshold chosen on validation folds only |
| `gate` | pre-registered rules; **exits 1** when nothing may be deployed; fails closed if MLflow is unreachable |
| `train_final` | fit on all participants, register MLflow pyfunc, set alias `@champion` |
| `report` | `reports/RESULTS.md` + per-participant chart |

`analysis/dvc.yaml` re-runs v1's random split on v2 features to measure the leakage.

## Quick start

```bash
# Python env (pipeline + tests)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps mediapipe==0.10.11

# Infrastructure secrets (random, written to .env, never printed)
python scripts/init_env.py

# Services
docker compose up -d --build                       # core stack
docker compose --profile airflow up -d --build     # + Airflow

# Pipeline (logs to the compose MLflow at localhost:5000)
bash scripts/run_pipeline.sh                       # = dvc repro with the venv on PATH

# End-to-end check: stream a recorded video through the WebSocket
python scripts/smoke_test_stream.py data/raw/drowsy/drowsy3.mov --seconds 90
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API docs | http://localhost:8000/docs |
| Model server docs | http://localhost:8001/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 (user/password in `.env`) |
| Prometheus / Alertmanager | http://localhost:9090 / http://localhost:9093 |
| Grafana | http://localhost:3001 |

## Serving behaviour

The backend answers every frame with a `status` message. States are fail-safe: an error or a
missing face is never displayed as "alert".

| State | Meaning |
|---|---|
| `warming_up` | the first window (length chosen in training) is not full yet |
| `alert` / `drowsy` | model prediction through an alarm state machine in seconds (on after 4 s, off after 6 s) |
| `no_face` | face missing ≥ 3 s (possible head drop / looking away) |
| `degraded` | input below 6 fps, or face visible in < 80 % of the window |
| `error` | model server unavailable or failing |

## CI (GitHub Actions)

`lint` → (`tests` with a 75 % coverage floor | `pipeline-config`: `dvc dag`, `dvc.lock`↔params
check, generated alert rules) → `docker-build` of all five images. No step is allowed to fail
silently. To make CI gate merges, protect `main` and mark these jobs as required checks.

## Project history (v1 → v2)

| | v1 (course submission) | v2 |
|---|---|---|
| Evaluation | random window split (same people in train and test) | nested leave-one-participant-out |
| Reported score | XGBoost F1 0.934 (optimistic) | best per-person ROC-AUC 0.86 (PERCLOS rule, 24 people) |
| Frames | 2 fps, rotation ignored (10/24 videos sideways), forced 640×480 | 10 fps by timestamp, rotation applied, real size |
| Data | `drowsy8` was a copy of `alert8` | original file restored; duplicate check in the data DAG |
| Settings | params.yaml **and** env defaults (window 2 vs 30) | params.yaml only; CI checks dvc.lock |
| Gate | logged, never blocked; "MLflow down" promoted | exits 1; fails closed; must beat a baseline |
| Serving errors | shown as "alert" | `error` / `no_face` / `degraded` states |
| CI | `\|\| true` on lint and tests | every step gates; 5 images built |

v1 numbers (27,783 windows, 96.4 % face detection, 134 tests, PySpark windowing) remain
reproducible at the `v1-course-submission` tag.

## Limitations

- 24 participants; labels are per video (self-reported KSS), recorded at a desk, not driving.
  Detecting a 0.05 AUC difference reliably would need on the order of 100+ people.
- Raw videos are not in the repo (UTA-RLDD licence): fetch a fold with
  `scripts/fetch_rldd_fold.py` and symlink it to `data/raw/fold<N>`.
- Sessions live in memory (one backend worker); scaling out needs shared session state.
- Airflow runs the SequentialExecutor on SQLite (demo setup).
