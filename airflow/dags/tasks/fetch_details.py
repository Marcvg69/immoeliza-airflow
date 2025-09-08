"""Airflow task: fetch property details for URLs parquet."""
from immoeliza.scraping.details import run as run_details

def run(**context):
    task_id = context["task"].task_id
    kind = "apartments" if "apartment" in task_id else "houses"
    conf = getattr(context.get("dag_run"), "conf", {}) or {}
    limit = conf.get("details_limit")  # optional: {"details_limit": 50}
    return run_details(kind, limit=limit)
