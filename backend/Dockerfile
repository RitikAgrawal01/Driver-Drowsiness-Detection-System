# Backend — FastAPI + WebSocket + MediaPipe. Build context: repo root (needs ddd_core).
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install --no-deps mediapipe==0.10.11

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
RUN useradd -m -u 1000 appuser && mkdir -p /app/logs/backend && chown -R appuser:appuser /app
COPY --chown=appuser:appuser ddd_core /app/ddd_core
COPY --chown=appuser:appuser backend /app/backend
USER appuser
ENV PYTHONPATH=/app DDD_PARAMS=/app/params.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# One worker: sessions live in memory (documented limitation; scale-out needs Redis + sticky WebSockets).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
