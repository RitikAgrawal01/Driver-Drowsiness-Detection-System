# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Driver Drowsiness Detection System
# Convenience commands for development workflow
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup install lint test docker-up docker-down mlflow airflow dvc-dag clean

# Default target
help:
	@echo ""
	@echo "  Driver Drowsiness Detection System — Make Commands"
	@echo "  ──────────────────────────────────────────────────"
	@echo "  make setup        Initialize git-lfs and dvc"
	@echo "  make install      Install all Python dependencies"
	@echo "  make lint         Run Black + Flake8"
	@echo "  make format       Auto-format with Black"
	@echo "  make test         Run all unit tests with coverage"
	@echo "  make docker-up    Start all core services"
	@echo "  make docker-down  Stop all services"
	@echo "  make airflow-up   Start Airflow separately"
	@echo "  make mlflow-ui    Open MLflow UI in browser"
	@echo "  make dvc-dag      Show DVC pipeline DAG"
	@echo "  make dvc-repro    Run full DVC pipeline"
	@echo "  make clean        Remove __pycache__, .pyc, logs"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
setup:
	@echo "→ Installing Git LFS..."
	git lfs install
	@echo "→ Initializing DVC..."
	dvc init --no-scm || true
	@echo "→ Copying .env template..."
	cp -n .env.template .env || echo ".env already exists, skipping."
	@echo "✓ Setup complete. Edit .env before running services."

# ── Install ───────────────────────────────────────────────────────────────
install:
	pip install -r backend/requirements.txt
	pip install -r model_server/requirements.txt
	pip install pytest pytest-cov black flake8

# ── Lint & Format ─────────────────────────────────────────────────────────
lint:
	black --check backend/ model_server/ airflow/ tests/
	flake8 backend/ model_server/ airflow/ tests/ --max-line-length=120

format:
	black backend/ model_server/ airflow/ tests/

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --cov=backend --cov=model_server --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

# ── Docker ────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build -d
	@echo "✓ Services started. Visit http://localhost:3000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

airflow-up:
	docker-compose --profile airflow up -d airflow
	@echo "✓ Airflow started at http://localhost:8080 (admin/admin)"

# ── MLflow ────────────────────────────────────────────────────────────────
mlflow-ui:
	@echo "→ MLflow UI: http://localhost:5000"
	docker-compose up -d mlflow

# ── DVC ───────────────────────────────────────────────────────────────────
dvc-dag:
	dvc dag

dvc-repro:
	dvc repro

dvc-push:
	dvc push

dvc-pull:
	dvc pull

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned."
