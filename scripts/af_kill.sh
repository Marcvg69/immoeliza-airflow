#!/usr/bin/env bash
set -euo pipefail
echo "Killing Airflow webserver/scheduler on 8080 + Dag Processor on 8793…"
lsof -ti tcp:8080 | xargs -r kill -9 || true
lsof -ti tcp:8793 | xargs -r kill -9 || true
pkill -f "airflow webserver|airflow scheduler|gunicorn|dag_processor" || true
echo "Done."
