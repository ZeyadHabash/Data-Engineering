import pandas as pd
from functions import extract_clean, transform, load_to_db
from fintech_dashboard import run_dashboard

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Define the DAG
default_args = {
    "owner": "Zeyad_Habash",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='fintech etl pipeline where we extract, clean, transform and load data to postgres',
)

with DAG(
    dag_id='fintech_etl_pipeline',
    # could be @daily, @hourly, etc or a cron expression '* * * * *'
    schedule_interval='@once',
    default_args=default_args,
    tags=['fintech-pipeline'],
)as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id='extract_clean',
        python_callable=extract_clean,
        op_kwargs={
            'filename': './data/fintech_data_49_52_16824.csv'
        }
    )

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
        op_kwargs={
            'filename': './data/fintech_clean.parquet'
        }
    )

    load_to_postgres_task = PythonOperator(
        task_id='load_to_db',
        python_callable=load_to_db,
        op_kwargs={
            'filename': './data/fintech_transformed.parquet'
        }
    )

    run_dashboard_task = BashOperator(
        task_id='run_dashboard',
        bash_command='streamlit run /opt/airflow/dags/fintech_dashboard.py -- --filename /opt/airflow/data/fintech_clean.parquet --lookup_table_filename /opt/airflow/data/lookup_table.csv',
        dag=dag,
    )

# Define the task dependencies
extract_clean_task >> transform_task >> load_to_postgres_task >> run_dashboard_task
