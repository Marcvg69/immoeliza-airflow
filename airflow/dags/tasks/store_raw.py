"""Airflow task: consolidate per-town CSVs into daily parquet."""
from immoeliza.scraping.consolidate import consolidate_urls

def run(**context):
    task_id = context["task"].task_id
    kind = "apartments" if "apartment" in task_id else "houses"
    return consolidate_urls(kind)
