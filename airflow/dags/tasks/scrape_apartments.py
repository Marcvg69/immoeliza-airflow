"""Airflow task: scrape apartments URLs."""
from immoeliza.scraping.apartments import run as run_scrape

def run(**context):
    return run_scrape()
