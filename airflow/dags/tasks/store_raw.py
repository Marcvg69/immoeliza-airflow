# airflow/dags/tasks/store_raw.py
from immoeliza.scraping.consolidate import consolidate_urls
def run(kind: str):
    return consolidate_urls(kind)
