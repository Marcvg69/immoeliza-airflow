"""Airflow task: scrape houses URLs."""
from immoeliza.scraping.houses import run as run_scrape

def run(**context):
    return run_scrape()
