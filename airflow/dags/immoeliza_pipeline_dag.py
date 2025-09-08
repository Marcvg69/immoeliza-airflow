# airflow/dags/immoeliza_pipeline_dag.py
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta

# Make our package importable inside Airflow workers
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from airflow import DAG
from airflow.operators.python import PythonOperator

# --- task callables (thin wrappers that live under airflow/dags/tasks) ---
from tasks.scrape_apartments import run as t_scrape_apartments
from tasks.scrape_houses import run as t_scrape_houses
from tasks.store_raw import run as t_store_raw
from tasks.fetch_details import run as t_fetch_details
from tasks.compact_details import run as t_compact_details
from tasks.clean_for_analysis import run as t_clean_for_analysis
from tasks.clean_for_training import run as t_clean_for_training
from tasks.train_regression import run as t_train_regression
from tasks.save_analytics import run as t_save_analytics
from tasks.save_model import run as t_save_model
from tasks.notify import run as t_notify

default_args = {
    "owner": "immo",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="immoeliza_pipeline",
    description="Immo-Eliza end-to-end ETL + model",
    start_date=datetime(2025, 9, 1),
    schedule_interval="0 3 * * *",  # daily @ 03:00
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["immoeliza"],
) as dag:

    # --- 1) URL scraping per kind ---
    scrape_apartments = PythonOperator(
        task_id="scrape_apartments",
        python_callable=t_scrape_apartments,
    )
    scrape_houses = PythonOperator(
        task_id="scrape_houses",
        python_callable=t_scrape_houses,
    )

    # --- 2) Consolidate URLs to daily parquet ---
    store_raw_apartments = PythonOperator(
        task_id="store_raw_apartments",
        python_callable=t_store_raw,
        op_kwargs={"kind": "apartments"},
    )
    store_raw_houses = PythonOperator(
        task_id="store_raw_houses",
        python_callable=t_store_raw,
        op_kwargs={"kind": "houses"},
    )

    # --- 3) Fetch details from consolidated URLs ---
    fetch_details_apartments = PythonOperator(
        task_id="fetch_details_apartments",
        python_callable=t_fetch_details,
        op_kwargs={"kind": "apartments", "limit": None},  # set limit to small int for smoke tests
    )
    fetch_details_houses = PythonOperator(
        task_id="fetch_details_houses",
        python_callable=t_fetch_details,
        op_kwargs={"kind": "houses", "limit": None},
    )

    # --- 4) Compact all detail CSVs to 1 parquet per kind ---
    compact_details_apartments = PythonOperator(
        task_id="compact_details_apartments",
        python_callable=t_compact_details,
        op_kwargs={"kind": "apartments"},
    )
    compact_details_houses = PythonOperator(
        task_id="compact_details_houses",
        python_callable=t_compact_details,
        op_kwargs={"kind": "houses"},
    )

    # --- 5) Analytics DuckDB + training parquet ---
    clean_for_analysis = PythonOperator(
        task_id="clean_for_analysis",
        python_callable=t_clean_for_analysis,
    )
    clean_for_training = PythonOperator(
        task_id="clean_for_training",
        python_callable=t_clean_for_training,
    )

    # --- 6) Train model (saves artifact) ---
    train_regression = PythonOperator(
        task_id="train_regression",
        python_callable=t_train_regression,
    )
    save_model = PythonOperator(
        task_id="save_model",
        python_callable=t_save_model,   # reads the artifact path or just logs
    )

    # --- 7) Save analytics + notify ---
    save_analytics = PythonOperator(
        task_id="save_analytics",
        python_callable=t_save_analytics,
    )
    notify = PythonOperator(
        task_id="notify",
        python_callable=t_notify,
        trigger_rule="all_done",
    )

    # ---- graph ----
    scrape_apartments >> store_raw_apartments >> fetch_details_apartments >> compact_details_apartments
    scrape_houses     >> store_raw_houses     >> fetch_details_houses     >> compact_details_houses

    [compact_details_apartments, compact_details_houses] >> clean_for_analysis >> save_analytics
    [compact_details_apartments, compact_details_houses] >> clean_for_training >> train_regression >> save_model

    [save_analytics, save_model] >> notify
