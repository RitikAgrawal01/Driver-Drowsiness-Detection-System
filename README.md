# Driver Drowsiness Detection System

Real-time driver drowsiness detection using MediaPipe facial landmarks, XGBoost/SVM classification, and a full end-to-end MLOps pipeline.

## Architecture Overview

```
React Frontend (port 3000)
        │  REST + WebSocket
FastAPI Backend (port 8000)
        │  REST
Model Server (port 8001)  ◄──── MLflow Registry (port 5000)
        │
   XGBoost / SVM model

Monitoring: Prometheus (9090) + Grafana (3001)
Pipeline:   Apache Airflow (8080) + PySpark + DVC
```

## Project Structure

```
Driver-Drowsiness-Detection-System/
├── frontend/           # React + Vite web application
├── backend/            # FastAPI main API + WebSocket server
├── model_server/       # MLflow model inference server
├── airflow/            # Airflow DAGs and plugins
│   └── dags/
├── data/               # DVC-tracked data (not in Git)
│   ├── raw/            # Raw video files
│   ├── landmarks/      # MediaPipe landmark CSVs
│   ├── features/       # Engineered feature CSVs + baseline.json
│   └── processed/      # Train/test splits
├── models/             # DVC-tracked model artifacts
├── monitoring/
│   ├── prometheus/     # prometheus.yml + alert_rules.yml
│   └── grafana/        # Dashboards + provisioning
├── tests/
│   ├── unit/
│   └── integration/
├── docs/               # All required documentation
├── .dvc/               # DVC internals (committed to Git)
├── dvc.yaml            # DVC pipeline definition
├── docker-compose.yml
├── .env.template
├── .gitignore
└── .gitattributes      # Git LFS tracking rules
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Git + Git LFS
- Python 3.11+
- Node.js 20+

### Setup

```bash
# 1. Clone and enter repo
git clone https://github.com/<your-username>/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System

# 2. Copy environment template
cp .env.template .env
# Edit .env with your values

# 3. Install Git LFS
git lfs install

# 4. Initialize DVC
dvc init
dvc remote add -d localremote /path/to/dvc_remote

# 5. Start core services
docker-compose up --build

# 6. Start Airflow (separate profile)
docker-compose --profile airflow up airflow
```

### Service URLs
| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000/docs |
| Model Server | http://localhost:8001/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

## Documentation
See `/docs` folder for:
- `architecture.md` — System architecture diagram and explanation
- `HLD.md` — High-level design document
- `LLD.md` — Low-level design with all API endpoint specifications
- `test_plan.md` — Test plan and test cases
- `user_manual.md` — Non-technical user guide

## Tech Stack
| Layer | Technology |
|---|---|
| Data Pipeline | Apache Airflow, PySpark, MediaPipe, OpenCV |
| Version Control | Git LFS, DVC |
| ML Training | XGBoost, SVM, scikit-learn |
| Experiment Tracking | MLflow |
| Backend API | FastAPI, WebSocket |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |
| Frontend | React, Vite |
