# ─────────────────────────────────────────────────────────────────
# Frontend — React + Vite → Nginx production server
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Build React app ─────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install deps first (cache layer)
COPY package.json package-lock.json* ./
# Note: npm ci requires a package-lock.json. If it fails, change this to: RUN npm install
RUN npm ci --silent 

# Build args injected at docker-compose build time
# MUST USE 127.0.0.1 INSTEAD OF localhost TO BYPASS WSL/WINDOWS IPv6 BUG
ARG VITE_BACKEND_URL=http://127.0.0.1:8000
ARG VITE_WS_URL=ws://127.0.0.1:8000
ARG VITE_GRAFANA_URL=http://127.0.0.1:3001
ARG VITE_MLFLOW_URL=http://127.0.0.1:5000
ARG VITE_AIRFLOW_URL=http://127.0.0.1:8080
ARG VITE_PROMETHEUS_URL=http://127.0.0.1:9090

ENV VITE_BACKEND_URL=$VITE_BACKEND_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_GRAFANA_URL=$VITE_GRAFANA_URL
ENV VITE_MLFLOW_URL=$VITE_MLFLOW_URL
ENV VITE_AIRFLOW_URL=$VITE_AIRFLOW_URL
ENV VITE_PROMETHEUS_URL=$VITE_PROMETHEUS_URL

COPY . .
RUN npm run build

# ── Stage 2: Nginx production server ─────────────────────────
FROM nginx:1.25-alpine AS runtime

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# SPA routing config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -q -O /dev/null http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]