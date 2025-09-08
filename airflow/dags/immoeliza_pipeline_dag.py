from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

def _noop():
    return "ok"

default_args = dict(owner="vincent", retries=1, retry_delay=timedelta(minutes=5))

with DAG(
    dag_id="immoeliza_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
    default_args=default_args,
    tags=["immoeliza"],
) as dag:

    start = EmptyOperator(task_id="start")

    # Placeholders - replace python_callable with real functions later
    scrape_apartments = PythonOperator(task_id="scrape_apartments", python_callable=_noop)
    store_raw_apartments = PythonOperator(task_id="store_raw_apartments", python_callable=_noop)

    scrape_houses = PythonOperator(task_id="scrape_houses", python_callable=_noop)
    store_raw_houses = PythonOperator(task_id="store_raw_houses", python_callable=_noop)

    raw_all_ready = EmptyOperator(task_id="raw_all_ready")

    clean_for_analysis = PythonOperator(task_id="clean_for_analysis", python_callable=_noop)
    save_analytics_duckdb = PythonOperator(task_id="save_analytics_duckdb", python_callable=_noop)

    clean_for_training = PythonOperator(task_id="clean_for_training", python_callable=_noop)
    save_training_dataset = PythonOperator(task_id="save_training_dataset", python_callable=_noop)
    train_regression = PythonOperator(task_id="train_regression", python_callable=_noop)
    save_model = PythonOperator(task_id="save_model", python_callable=_noop)

    finished = EmptyOperator(task_id="finished")

    start >> [scrape_apartments, scrape_houses]
    scrape_apartments >> store_raw_apartments >> raw_all_ready
    scrape_houses >> store_raw_houses >> raw_all_ready
    raw_all_ready >> [clean_for_analysis, clean_for_training]
    clean_for_analysis >> save_analytics_duckdb
    clean_for_training >> save_training_dataset >> train_regression >> save_model >> finished
