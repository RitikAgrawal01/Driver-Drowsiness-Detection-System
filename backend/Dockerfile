# ─────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
    --default-timeout=1000 \
    --prefer-binary \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    mediapipe==0.10.11 --no-deps && \
    pip install --no-cache-dir --prefix=/install \
    --default-timeout=1000 \
    --prefer-binary \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .

# USER appuser

COPY --chown=appuser:appuser . /app/backend

EXPOSE 8000

# Health check uses the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
