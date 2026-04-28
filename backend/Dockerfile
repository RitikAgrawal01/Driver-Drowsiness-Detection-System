# ─────────────────────────────────────────────────────────────────
# Backend — FastAPI + MediaPipe + WebSocket
# Multi-stage build: builder installs deps, runtime is slim
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System libs needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install OpenCV first (standard)
RUN pip install --no-cache-dir --prefix=/install opencv-python-headless==4.9.0.80

# ── THE HYBRID FIX ──
# 1. Use wildcards so COPY doesn't fail if files are missing (GitHub Actions case)
COPY mediapipe-*.whl* torch-*.whl* ./

# 2. Install MediaPipe: Local Wheel OR PyPI
RUN if [ -f mediapipe-0.10.11-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ]; then \
        pip install --no-cache-dir --no-deps --prefix=/install mediapipe-0.10.11-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl; \
    else \
        pip install --no-cache-dir --prefix=/install mediapipe==0.10.11; \
    fi

# 3. Install PyTorch: Local Wheel OR PyPI
RUN if [ -f torch-2.11.0-cp311-cp311-manylinux_2_28_x86_64.whl ]; then \
        pip install --no-cache-dir --no-deps --prefix=/install torch-2.11.0-cp311-cp311-manylinux_2_28_x86_64.whl; \
    else \
        pip install --no-cache-dir --prefix=/install torch==2.1.0; \
    fi

# 4. Handle remaining requirements
COPY requirements.txt .
# Remove mediapipe/torch from requirements if they exist to avoid double-install
RUN sed -i '/mediapipe/d' requirements.txt && sed -i '/torch/d' requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs/backend && \
    chown -R appuser:appuser /app

# Copy application source into a preserved package directory
COPY --chown=appuser:appuser . /app/backend

USER appuser

# Tell Python where to find the 'backend' package
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Tell uvicorn to run from inside the backend package
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]