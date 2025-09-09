# airflow/dags/smoke_dag.py
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

with DAG("smoke", start_date=datetime(2024,1,1), schedule=None, catchup=False) as dag:
    EmptyOperator(task_id="ok")

