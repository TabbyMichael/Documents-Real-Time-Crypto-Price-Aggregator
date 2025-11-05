# scheduler/prefect_flow.py
from prefect import flow, task
from etl.pipeline import run_pipeline

@task
def run_etl_task():
    run_pipeline()

@flow
def crypto_flow():
    run_etl_task()

if __name__ == '__main__':
    crypto_flow()