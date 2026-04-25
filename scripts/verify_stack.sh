#!/usr/bin/env bash
# scripts/verify_stack.sh
# ─────────────────────────────────────────────────────────────────
# Run after docker compose up to verify all services are healthy.
# Usage: ./scripts/verify_stack.sh
# ─────────────────────────────────────────────────────────────────

set -euo pipefail
GREEN='\033[0;32m'; RED='\033[0;31m'; AMBER='\033[0;33m'
CYAN='\033[0;36m'; RESET='\033[0m'; BOLD='\033[1m'

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; FAILURES=$((FAILURES+1)); }
warn() { echo -e "  ${AMBER}!${RESET} $1"; }
info() { echo -e "  ${CYAN}→${RESET} $1"; }

FAILURES=0

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   DDD Stack Verification                                 ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Docker container status ────────────────────────────────
echo -e "${BOLD}── Container Status ────────────────────────────────────────${RESET}"
CONTAINERS=(ddd_frontend ddd_backend ddd_model_server ddd_mlflow ddd_prometheus ddd_grafana)
for name in "${CONTAINERS[@]}"; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "not found")
    RUNNING=$(docker inspect --format='{{.State.Running}}' "$name" 2>/dev/null || echo "false")
    if [ "$RUNNING" = "true" ]; then
        if [ "$STATUS" = "healthy" ]; then
            ok "$name — running & healthy"
        elif [ "$STATUS" = "starting" ]; then
            warn "$name — running, health check starting..."
        else
            warn "$name — running (health: $STATUS)"
        fi
    else
        fail "$name — NOT running"
    fi
done

echo ""
echo -e "${BOLD}── HTTP Endpoint Checks ────────────────────────────────────${RESET}"

check_url() {
    local NAME="$1" URL="$2" EXPECTED="$3"
    local STATUS
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL" 2>/dev/null || echo "000")
    if [ "$STATUS" = "$EXPECTED" ] || { [ "$EXPECTED" = "2xx" ] && [[ "$STATUS" =~ ^2 ]]; }; then
        ok "$NAME ($URL) → $STATUS"
    else
        fail "$NAME ($URL) → $STATUS (expected $EXPECTED)"
    fi
}

# Changed to 127.0.0.1 to avoid WSL IPv6 curl hangs
check_url "Frontend"             "http://127.0.0.1:3000"        "200"
check_url "Backend /health"      "http://127.0.0.1:8000/health"  "200"
check_url "Backend /metrics"     "http://127.0.0.1:8000/metrics" "200"
check_url "Backend /docs"        "http://127.0.0.1:8000/docs"    "200"
check_url "Model Server /health" "http://127.0.0.1:8001/health" "200"
check_url "Model Server /metrics" "http://127.0.0.1:8001/metrics" "200"
check_url "MLflow"               "http://127.0.0.1:5000/"  "200"
check_url "Prometheus"           "http://127.0.0.1:9090/-/ready" "200"
check_url "Grafana"              "http://127.0.0.1:3001/api/health" "200"

echo ""
echo -e "${BOLD}── Prometheus Metrics Check ────────────────────────────────${RESET}"

METRICS_BODY=$(curl -s --max-time 5 "http://127.0.0.1:8000/metrics" 2>/dev/null || echo "")
REQUIRED_METRICS=(
    "ddd_inference_latency_seconds"
    "ddd_prediction_confidence"
    "ddd_feature_drift_score"
    "ddd_overall_drift_score"
    "ddd_requests_total"
    "ddd_active_sessions"
    "ddd_drowsy_alerts_total"
    "ddd_frames_processed_total"
    "ddd_model_server_reachable"
)
for metric in "${REQUIRED_METRICS[@]}"; do
    if echo "$METRICS_BODY" | grep -q "$metric"; then
        ok "Metric: $metric"
    else
        fail "Metric missing: $metric"
    fi
done

echo ""
echo -e "${BOLD}── Prometheus Scrape Targets ───────────────────────────────${RESET}"

TARGETS=$(curl -s --max-time 5 "http://127.0.0.1:9090/api/v1/targets" 2>/dev/null || echo "{}")
for job in "ddd_backend" "ddd_model_server"; do
    if echo "$TARGETS" | grep -q "\"$job\""; then
        HEALTH=$(echo "$TARGETS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
targets = data.get('data', {}).get('activeTargets', [])
for t in targets:
    if t.get('labels', {}).get('job') == '$job':
        print(t.get('health', 'unknown'))
        break
" 2>/dev/null || echo "unknown")
        if [ "$HEALTH" = "up" ]; then
            ok "Prometheus scraping: $job (up)"
        else
            warn "Prometheus scraping: $job ($HEALTH)"
        fi
    else
        fail "Prometheus target not found: $job"
    fi
done

echo ""
echo -e "${BOLD}── Model Server Readiness ──────────────────────────────────${RESET}"

READY=$(curl -s --max-time 5 "http://127.0.0.1:8001/ready" 2>/dev/null || echo "{}")
MODEL_LOADED=$(echo "$READY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','?'))" 2>/dev/null || echo "?")
if [ "$MODEL_LOADED" = "True" ] || [ "$MODEL_LOADED" = "true" ]; then
    MODEL_VERSION=$(echo "$READY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_version','?'))" 2>/dev/null || echo "?")
    ALGORITHM=$(echo "$READY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('algorithm','?'))" 2>/dev/null || echo "?")
    ok "Model loaded: $ALGORITHM v$MODEL_VERSION"
else
    warn "Model not yet loaded (may still be starting — check: docker compose logs model_server)"
fi

echo ""
echo -e "${BOLD}── Summary ─────────────────────────────────────────────────${RESET}"

if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ALL CHECKS PASSED ✓${RESET}"
    echo ""
    echo "  Service URLs:"
    echo -e "    Frontend    ${CYAN}http://127.0.0.1:3000${RESET}"
    echo -e "    API Docs    ${CYAN}http://127.0.0.1:8000/docs${RESET}"
    echo -e "    MLflow      ${CYAN}http://127.0.0.1:5000${RESET}"
    echo -e "    Grafana     ${CYAN}http://127.0.0.1:3001${RESET}  (admin / admin)"
    echo -e "    Prometheus  ${CYAN}http://127.0.0.1:9090${RESET}"
    echo ""
    echo "  Next steps:"
    echo "    Warm up:  python tools/send_test_traffic.py --duration 60"
    echo "    Drift:    python tools/simulate_drift.py --mode lighting --count 300"
else
    echo -e "${RED}${BOLD}  $FAILURES CHECK(S) FAILED ✗${RESET}"
    echo ""
    echo "  Debug commands:"
    echo "    docker compose ps"
    echo "    docker compose logs backend"
    echo "    docker compose logs model_server"
    echo "    docker compose restart backend"
fi
echo ""