# Model server — FastAPI + MLflow pyfunc champion. Build context: repo root.
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY model_server/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
# libgomp1: XGBoost OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser ddd_core /app/ddd_core
COPY --chown=appuser:appuser model_server /app/model_server
USER appuser
ENV PYTHONPATH=/app DDD_PARAMS=/app/params.yaml
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1
CMD ["uvicorn", "model_server.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
