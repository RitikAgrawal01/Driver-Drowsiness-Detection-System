# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Driver Drowsiness Detection System
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup install lint format test docker-up docker-down docker-build \
        airflow-up mlflow-ui dvc-dag dvc-repro drift-sim verify clean logs

help:
	@echo ""
	@echo "  Driver Drowsiness Detection System"
	@echo "  ═══════════════════════════════════════════════════════"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup          One-time: Git LFS + DVC + .env"
	@echo "    make install        Install Python dev dependencies"
	@echo ""
	@echo "  CODE QUALITY"
	@echo "    make lint           Black check + Flake8"
	@echo "    make format         Auto-format with Black"
	@echo "    make test           Run all unit tests with coverage"
	@echo ""
	@echo "  DOCKER"
	@echo "    make docker-up      Start all core services"
	@echo "    make docker-down    Stop all services"
	@echo "    make docker-build   Rebuild all images"
	@echo "    make airflow-up     Start Airflow (separate profile)"
	@echo "    make verify         Verify stack health after startup"
	@echo "    make logs           Tail backend + model_server logs"
	@echo ""
	@echo "  ML PIPELINE"
	@echo "    make train          Train XGBoost + SVM + evaluate"
	@echo "    make dvc-dag        Show DVC pipeline DAG"
	@echo "    make dvc-repro      Run full DVC pipeline"
	@echo "    make mlflow-ui      Open MLflow UI in browser"
	@echo ""
	@echo "  MONITORING"
	@echo "    make traffic        Send 2min of test traffic to backend"
	@echo "    make drift-sim      Simulate lighting drift scenario"
	@echo ""
	@echo "  CLEANUP"
	@echo "    make clean          Remove __pycache__, .pyc files"
	@echo "    make clean-docker   Remove stopped containers + dangling images"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────
setup:
	@echo "→ Installing Git LFS..."
	git lfs install
	@echo "→ Initialising DVC..."
	dvc init --no-scm 2>/dev/null || true
	mkdir -p ~/dvc_remote
	dvc remote add -f -d localremote ~/dvc_remote
	@echo "→ Copying .env template..."
	cp -n .env.template .env || echo ".env already exists"
	@echo "→ Creating data directories..."
	mkdir -p data/raw/drowsy data/raw/alert data/frames data/landmarks \
	         data/features data/processed models reports \
	         logs/backend logs/model_server
	@echo "✓ Setup complete. Edit .env before running docker-compose."

# ── Install ────────────────────────────────────────────────────────────────
install:
	pip install -r requirements-dev.txt

# ── Code quality ──────────────────────────────────────────────────────────
lint:
	black --check backend/ model_server/ airflow/scripts/ tests/
	flake8 backend/ model_server/ airflow/scripts/ tests/ \
	    --max-line-length=120 --ignore=W503,E303 --exclude=__pycache__

format:
	black backend/ model_server/ airflow/scripts/ tests/

test:
	pytest tests/unit/ -v --tb=short \
	    --cov=backend --cov=model_server \
	    --cov-report=term-missing \
	    --cov-report=xml:coverage.xml

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	@./scripts/start.sh

docker-down:
	docker compose down
	@echo "✓ All services stopped"

docker-build:
	docker compose build --no-cache

airflow-up:
	docker compose --profile airflow up -d airflow
	@echo "✓ Airflow started at http://localhost:8080 (admin/admin)"

verify:
	@bash scripts/verify_stack.sh

logs:
	docker compose logs -f backend model_server

# ── ML Pipeline ───────────────────────────────────────────────────────────
train:
	@echo "→ Checking if Docker MLflow is running on port 5000..."
	@curl -s http://localhost:5000 > /dev/null || (echo "❌ MLflow is not running. Run 'make docker-up' first." && exit 1)
	@echo "→ Training XGBoost..."
	export MLFLOW_TRACKING_URI=http://localhost:5000 && python model_server/train_xgboost.py
	@echo "→ Training SVM..."
	export MLFLOW_TRACKING_URI=http://localhost:5000 && python model_server/train_svm.py
	@echo "→ Evaluating and promoting winner..."
	export MLFLOW_TRACKING_URI=http://localhost:5000 && python model_server/evaluate.py
	@echo "✓ Training complete. Open MLflow at http://localhost:5000"

dvc-dag:
	dvc dag

dvc-repro:
	dvc repro

mlflow-ui:
	@echo "→ Opening MLflow UI at http://localhost:5000"
	@command -v xdg-open &>/dev/null && xdg-open http://localhost:5000 || \
	 command -v open &>/dev/null && open http://localhost:5000 || \
	 echo "Open http://localhost:5000 in your browser"

# ── Monitoring ────────────────────────────────────────────────────────────
traffic:
	python tools/send_test_traffic.py --duration 120 --verbose

drift-sim:
	python tools/simulate_drift.py --mode lighting --count 300 --verbose

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

clean-docker:
	docker compose down --remove-orphans
	docker image prune -f
	@echo "✓ Docker cleaned"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Retraining Pipeline commands
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: retrain retrain-force webhook webhook-trigger retrain-status model-reload

# Start the Prometheus → Airflow webhook bridge
webhook:
	@echo "→ Starting Prometheus webhook bridge on port 9095..."
	python tools/prometheus_webhook.py --port 9095 \
	    --airflow-url http://127.0.0.1:8080

# Manually trigger retraining via webhook bridge (demo command)
webhook-trigger:
	@echo "→ Manually triggering retraining DAG..."
	curl -X POST http://127.0.0.1:9095/trigger \
	    -H "Content-Type: application/json" \
	    -d '{"reason": "manual_demo_trigger"}' | python3 -m json.tool

# Trigger retraining DAG directly via Airflow API (no webhook bridge needed)
retrain:
	@echo "→ Triggering ddd_retrain_pipeline DAG..."
	curl -X POST http://127.0.0.1:8080/api/v1/dags/ddd_retrain_pipeline/dagRuns \
	    -H "Content-Type: application/json" \
	    -u admin:admin \
	    -d '{"dag_run_id": "manual__$(shell date +%Y%m%dT%H%M%S)", "conf": {"retrain_reason": "manual"}}' \
	    | python3 -m json.tool

# Force retrain (skip drift/F1 threshold checks)
retrain-force:
	@echo "→ Force-triggering ddd_retrain_pipeline DAG (skipping threshold checks)..."
	curl -X POST http://127.0.0.1:8080/api/v1/dags/ddd_retrain_pipeline/dagRuns \
	    -H "Content-Type: application/json" \
	    -u admin:admin \
	    -d '{"dag_run_id": "force__$(shell date +%Y%m%dT%H%M%S)", "conf": {"force_retrain": true}}' \
	    | python3 -m json.tool

# Check status of the last retraining run
retrain-status:
	@echo "→ Last 3 ddd_retrain_pipeline runs:"
	curl -s http://127.0.0.1:8080/api/v1/dags/ddd_retrain_pipeline/dagRuns?limit=3 \
	    -u admin:admin | python3 -m json.tool

# Reload model server after manual promotion (if hot-reload not available)
model-reload:
	@echo "→ Restarting model server to load new Production model..."
	docker compose restart model_server
	@echo "→ Waiting for model server to be healthy..."
	sleep 20
	curl -s http://127.0.0.1:8001/ready | python3 -m json.tool