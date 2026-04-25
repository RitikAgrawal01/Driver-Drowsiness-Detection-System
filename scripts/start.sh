#!/usr/bin/env bash
# scripts/start.sh
# ─────────────────────────────────────────────────────────────────
# One-command startup script for the DDD system.
# Builds images, starts all core services, waits for health,
# then opens the browser.
#
# Usage:
#   ./scripts/start.sh           # core services only
#   ./scripts/start.sh --airflow # include Airflow
#   ./scripts/start.sh --build   # force rebuild images
# ─────────────────────────────────────────────────────────────────

set -euo pipefail
GREEN='\033[0;32m'; CYAN='\033[0;36m'; AMBER='\033[0;33m'; RESET='\033[0m'; BOLD='\033[1m'

WITH_AIRFLOW=false
FORCE_BUILD=false

for arg in "$@"; do
    case $arg in
        --airflow) WITH_AIRFLOW=true ;;
        --build)   FORCE_BUILD=true ;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Driver Drowsiness Detection System — Starting          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Check .env exists ─────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo -e "${AMBER}  ! .env not found — copying from .env.template${RESET}"
    cp .env.template .env
    echo -e "${AMBER}  ! Please edit .env and set AIRFLOW__CORE__FERNET_KEY${RESET}"
    echo -e "    python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    echo ""
fi

# ── Check models exist ────────────────────────────────────────
if [ ! -f "models/xgboost_model.pkl" ]; then
    echo -e "${AMBER}  ! No trained models found in models/${RESET}"
    echo -e "    Run Phase 4 training first via the Makefile wrapper:"
    echo -e "    ${CYAN}make train${RESET}"
    echo ""
fi

# ── Build ─────────────────────────────────────────────────────
BUILD_FLAG=""
[ "$FORCE_BUILD" = true ] && BUILD_FLAG="--build"

echo -e "${CYAN}→ Starting core services...${RESET}"
if [ "$WITH_AIRFLOW" = true ]; then
    docker compose --profile airflow up -d $BUILD_FLAG
else
    docker compose up -d $BUILD_FLAG backend model_server mlflow prometheus grafana frontend
fi

# ── Wait for backend health ───────────────────────────────────
echo -e "${CYAN}→ Waiting for backend to be healthy...${RESET}"
RETRIES=0
# Changed to 127.0.0.1 to avoid WSL IPv6 resolution hangs
until curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; do
    RETRIES=$((RETRIES+1))
    if [ $RETRIES -ge 30 ]; then
        echo -e "${AMBER}  Backend taking longer than expected. Check: docker compose logs backend${RESET}"
        break
    fi
    printf "."
    sleep 2
done
echo ""

# ── Summary ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  Stack started! Service URLs:${RESET}"
# Changed all URLs to 127.0.0.1 so you can safely click them in WSL
echo -e "  ${CYAN}Frontend    http://127.0.0.1:3000${RESET}"
echo -e "  ${CYAN}API Docs    http://127.0.0.1:8000/docs${RESET}"
echo -e "  ${CYAN}MLflow      http://127.0.0.1:5000${RESET}"
echo -e "  ${CYAN}Grafana     http://127.0.0.1:3001${RESET}  (admin / admin)"
echo -e "  ${CYAN}Prometheus  http://127.0.0.1:9090${RESET}"
[ "$WITH_AIRFLOW" = true ] && echo -e "  ${CYAN}Airflow     http://127.0.0.1:8080${RESET}  (admin / admin)"
echo ""
echo -e "  Verify: ${CYAN}./scripts/verify_stack.sh${RESET}"
echo -e "  Logs:   ${CYAN}docker compose logs -f backend${RESET}"
echo -e "  Stop:   ${CYAN}docker compose down${RESET}"
echo ""