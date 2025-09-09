"""
Airflow DAG: ImmoEliza end-to-end

Pipeline:
  1) URL scraping (apartments, houses)
  2) Consolidate URLs per kind
  3) Fetch details per kind
  4) Compact details (parquet) per kind
  5) Analytics cleaning -> analytics.duckdb + processed parquet
  6) Training cleaning -> processed training parquet
  7) Dual-target training:
       - price
       - price_per_m2 (if available; otherwise no-op)
Notes:
- Tasks return important artifact paths via XCom.
- Env knobs:
    IMMO_HEADLESS       -> "true"/"false" (used by the detail scraper underneath)
    IMMO_MAX_PAGES      -> number of search pages per town (default 1)
    IMMO_DETAILS_LIMIT  -> cap details per run (default None)
    IMMO_ANALYTICS_DB   -> path to DuckDB (default analytics/immoeliza.duckdb)
"""

from __future__ import annotations

import os
import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# --------- Thin wrappers so operators have stable callables ---------

def _scrape_apartments(**_):
    max_pages = int(os.getenv("IMMO_MAX_PAGES", "1"))
    from immoeliza.scraping.apartments import run as run_apts
    return run_apts(max_pages=max_pages)

def _scrape_houses(**_):
    max_pages = int(os.getenv("IMMO_MAX_PAGES", "1"))
    from immoeliza.scraping.houses import run as run_houses
    return run_houses(max_pages=max_pages)

def _consolidate(kind: str, **_):
    from immoeliza.scraping.consolidate import consolidate_urls
    return consolidate_urls(kind)

def _fetch_details(kind: str, **_):
    # Limit comes from env (string or empty)
    lim_raw = os.getenv("IMMO_DETAILS_LIMIT", "").strip()
    limit = int(lim_raw) if lim_raw.isdigit() else None
    from immoeliza.scraping.details import run as run_details
    return run_details(kind, limit=limit)

def _compact_details(kind: str, **_):
    from immoeliza.scraping.details_compact import compact
    return compact(kind)

def _clean_analysis(**_):
    from immoeliza.cleaning.analysis_clean import run as run_analysis
    return run_analysis()

def _clean_training(**_):
    from immoeliza.cleaning.training_clean import run as run_training
    return run_training()

def _train_price(ti, **_):
    """Train model to predict absolute price (EUR), leak-safe."""
    from immoeliza.modeling.train_regression import train
    training_file = ti.xcom_pull(task_ids="clean_training")
    # prefer explicit env override, else call as 'price'
    target = os.getenv("IMMO_TRAIN_TARGET", "price")
    return train(training_file, target=target)

def _train_price_per_m2(ti, **_):
    """Train model to predict price_per_m2 if that column exists and is usable."""
    from immoeliza.modeling.train_regression import train
    training_file = ti.xcom_pull(task_ids="clean_training")
    return train(training_file, target="price_per_m2")

# --------- DAG definition ---------

with DAG(
    dag_id="immoeliza_pipeline",
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Brussels"),
    schedule="0 6 * * *",  # daily at 06:00
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "immoeliza",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["immo", "scraping", "etl", "ml"],
) as dag:

    # 1) URL scraping
    scrape_apartments = PythonOperator(
        task_id="scrape_apartments",
        python_callable=_scrape_apartments,
    )
    scrape_houses = PythonOperator(
        task_id="scrape_houses",
        python_callable=_scrape_houses,
    )

    # 2) Consolidate
    consolidate_apartments = PythonOperator(
        task_id="consolidate_apartments",
        python_callable=_consolidate,
        op_kwargs={"kind": "apartments"},
    )
    consolidate_houses = PythonOperator(
        task_id="consolidate_houses",
        python_callable=_consolidate,
        op_kwargs={"kind": "houses"},
    )

    # 3) Fetch details
    fetch_details_apartments = PythonOperator(
        task_id="fetch_details_apartments",
        python_callable=_fetch_details,
        op_kwargs={"kind": "apartments"},
    )
    fetch_details_houses = PythonOperator(
        task_id="fetch_details_houses",
        python_callable=_fetch_details,
        op_kwargs={"kind": "houses"},
    )

    # 4) Compact details
    compact_details_apartments = PythonOperator(
        task_id="compact_details_apartments",
        python_callable=_compact_details,
        op_kwargs={"kind": "apartments"},
    )
    compact_details_houses = PythonOperator(
        task_id="compact_details_houses",
        python_callable=_compact_details,
        op_kwargs={"kind": "houses"},
    )

    # 5) Analytics clean + upsert DuckDB
    clean_analysis = PythonOperator(
        task_id="clean_analysis",
        python_callable=_clean_analysis,
    )

    # 6) Training clean
    clean_training = PythonOperator(
        task_id="clean_training",
        python_callable=_clean_training,
    )

    # 7) Dual-target training (run in parallel after training parquet exists)
    train_price = PythonOperator(
        task_id="train_price",
        python_callable=_train_price,
        provide_context=True,
    )
    train_price_per_m2 = PythonOperator(
        task_id="train_price_per_m2",
        python_callable=_train_price_per_m2,
        provide_context=True,
    )

    # ---- Wiring ----
    scrape_apartments >> consolidate_apartments >> fetch_details_apartments >> compact_details_apartments
    scrape_houses     >> consolidate_houses     >> fetch_details_houses     >> compact_details_houses

    [compact_details_apartments, compact_details_houses] >> clean_analysis >> clean_training >> [train_price, train_price_per_m2]
