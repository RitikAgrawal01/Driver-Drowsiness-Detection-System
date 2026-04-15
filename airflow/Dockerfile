FROM apache/airflow:2.9.1-python3.11

USER root

# Install Java (required by PySpark) and system libs for MediaPipe/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

COPY requirements.txt /tmp/airflow_requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow_requirements.txt
