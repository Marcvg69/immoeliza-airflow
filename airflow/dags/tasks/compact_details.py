# airflow/dags/tasks/compact_details.py
from immoeliza.scraping.details_compact import compact
def run(kind: str):
    return compact(kind)
