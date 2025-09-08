# airflow/dags/immoeliza_pipeline_dag.py
"""
ImmoEliza end-to-end pipeline DAG.

Per kind (apartments & houses):
  scrape_*  →  store_raw_*  →  fetch_*_details  →  compact_*_details
Both kinds then fan-in to:
  clean_for_analysis  →  clean_for_training  →  train_regression  →  notify
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

# ---- Task wrappers (kept tiny; live in airflow/dags/tasks/) ----
import tasks.scrape_apartments as tsa
import tasks.scrape_houses as tsh
import tasks.store_raw as tsr
import tasks.fetch_details as tfd
import tasks.compact_details as tcd

# NEW: cleaning + training
import tasks.clean_for_analysis as tca
import tasks.clean_for_training as tct
import tasks.train_regression as ttr

# Final notification (placeholder)
import tasks.notify as tn


DEFAULT_ARGS = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="immoeliza_pipeline",
    description="Scrape → Consolidate → Details → Compact → Clean → Train",
    start_date=datetime(2025, 9, 1),
    schedule_interval=None,   # set a cron later (e.g., "0 3 * * *")
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["immoeliza", "real-estate"],
    params={
        # Set during manual trigger to limit details scrape size (0 = unlimited)
        "details_limit": Param(0, type="integer", minimum=0),
        # Optional override for train task; otherwise it picks latest training.parquet
        "training_path": Param("", type="string"),
    },
) as dag:

    start = EmptyOperator(task_id="start")

    # -------- Apartments branch --------
    scrape_apartments = PythonOperator(
        task_id="scrape_apartments",
        python_callable=tsa.run,
    )

    store_raw_apartments = PythonOperator(
        task_id="store_raw_apartments",
        python_callable=tsr.run,
    )

    fetch_apartment_details = PythonOperator(
        task_id="fetch_apartment_details",
        python_callable=tfd.run,   # reads params.details_limit if provided
    )

    compact_apartment_details = PythonOperator(
        task_id="compact_apartment_details",
        python_callable=tcd.run,
    )

    # -------- Houses branch --------
    scrape_houses = PythonOperator(
        task_id="scrape_houses",
        python_callable=tsh.run,
    )

    store_raw_houses = PythonOperator(
        task_id="store_raw_houses",
        python_callable=tsr.run,
    )

    fetch_house_details = PythonOperator(
        task_id="fetch_house_details",
        python_callable=tfd.run,   # reads params.details_limit if provided
    )

    compact_house_details = PythonOperator(
        task_id="compact_house_details",
        python_callable=tcd.run,
    )

    # -------- NEW: Cleaning & Training fan-in --------
    clean_for_analysis = PythonOperator(
        task_id="clean_for_analysis",
        python_callable=tca.run,
    )

    clean_for_training = PythonOperator(
        task_id="clean_for_training",
        python_callable=tct.run,
    )

    train_regression = PythonOperator(
        task_id="train_regression",
        python_callable=ttr.run,
    )

    done = PythonOperator(
        task_id="notify",
        python_callable=tn.run,
    )

    # -------- Dependencies --------
    start >> [scrape_apartments, scrape_houses]

    scrape_apartments >> store_raw_apartments >> fetch_apartment_details >> compact_apartment_details
    scrape_houses     >> store_raw_houses     >> fetch_house_details     >> compact_house_details

    [compact_apartment_details, compact_house_details] >> clean_for_analysis
    clean_for_analysis >> clean_for_training >> train_regression >> done
