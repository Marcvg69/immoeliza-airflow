#!/usr/bin/env bash
set -euo pipefail

echo "Killing Airflow/gunicorn/webserver/scheduler/dag_processor…"
pkill -f "airflow webserver|airflow scheduler|dag_processor|gunicorn" 2>/dev/null || true

echo "Freeing ports 8080 and 8793 (if any)…"
(lsof -ti tcp:8080 || true) | xargs -r kill -9
(lsof -ti tcp:8793 || true) | xargs -r kill -9

echo "Done."