# Airflow — orchestrates the DVC pipeline (`dvc repro`); the repo is mounted at /opt/project.
# Build context: repo root.
FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libgl1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

USER airflow
COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps mediapipe==0.10.11
