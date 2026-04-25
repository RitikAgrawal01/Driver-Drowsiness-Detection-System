# ─────────────────────────────────────────────────────────────────
# Airflow — with Java (PySpark), MediaPipe, OpenCV, DVC
# Based on official Airflow image
# ─────────────────────────────────────────────────────────────────

FROM apache/airflow:2.9.1-python3.11

USER root

# Java (required by PySpark) + system libs for MediaPipe/OpenCV/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER airflow

COPY requirements.txt /tmp/airflow_requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow_requirements.txt