#!/usr/bin/env bash
# Airflow container start: migrate the DB, create the admin user from .env (once), run scheduler + webserver.
set -euo pipefail
airflow db migrate
if ! airflow users list 2>/dev/null | grep -qw "${AIRFLOW_ADMIN_USER}"; then
  airflow users create --username "${AIRFLOW_ADMIN_USER}" --password "${AIRFLOW_ADMIN_PASSWORD}" \
    --firstname DDD --lastname Admin --role Admin --email admin@example.com
fi
airflow scheduler &
exec airflow webserver
