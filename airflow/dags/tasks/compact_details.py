"""Airflow task: compact details CSVs into a daily parquet artifact."""
from immoeliza.scraping.details_compact import compact

def run(**context):
    task_id = context["task"].task_id
    kind = "apartments" if "apartment" in task_id else "houses"
    return compact(kind)
