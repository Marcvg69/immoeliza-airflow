# airflow/dags/tasks/fetch_details.py
from immoeliza.scraping.details import run as fetch_run
def run(kind: str, limit: int | None = None):
    return fetch_run(kind, limit=limit)

