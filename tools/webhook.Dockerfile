# Alertmanager -> Airflow webhook bridge. Build context: repo root.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pyyaml==6.0.3 && useradd -m -u 1000 appuser
COPY --chown=appuser:appuser tools/prometheus_webhook.py /app/prometheus_webhook.py
USER appuser
ENV DDD_PARAMS=/app/params.yaml
EXPOSE 9095
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9095/health')"
CMD ["python", "/app/prometheus_webhook.py"]
