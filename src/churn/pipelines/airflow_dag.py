"""Airflow DAG that orchestrates the full churn pipeline:

generate/refresh data -> validate with Great Expectations -> load to
Postgres -> build features -> train models (sklearn + PyTorch).

To use: copy or symlink this file into your $AIRFLOW_HOME/dags folder,
and make sure the `churn` package is importable (e.g. `pip install -e .`
in the same environment Airflow runs in).
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

default_args = {
    "owner": "data-team",
    "retries": 1,
}


@dag(
    dag_id="churn_pipeline",
    description="End-to-end customer churn data + training pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "ml"],
)
def churn_pipeline():
    @task
    def generate_data() -> str:
        from churn.data.generate_data import main as generate_main

        path = generate_main()
        return str(path)

    @task
    def validate_data(_: str) -> bool:
        from churn.data.validate_data import validate

        ok = validate()
        if not ok:
            raise ValueError("Data validation failed - stopping pipeline.")
        return ok

    @task
    def load_to_postgres(_: bool) -> int:
        from churn.data.ingest import load_csv_to_postgres

        return load_csv_to_postgres()

    @task
    def build_features(_: int) -> str:
        from churn.features.build_features import main as build_main

        build_main()
        return "features_built"

    @task
    def train_sklearn_model(_: str) -> str:
        from churn.models.train_sklearn import train

        return train()

    @task
    def train_torch_model(_: str) -> str:
        from churn.models.train_torch import train

        return train()

    raw_path = generate_data()
    validated = validate_data(raw_path)
    loaded = load_to_postgres(validated)
    features_ready = build_features(loaded)
    train_sklearn_model(features_ready)
    train_torch_model(features_ready)


churn_pipeline()
