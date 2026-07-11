# Architecture

```
                ┌─────────────────┐
                │  generate_data   │  synthetic customer records
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │  validate_data   │  Great Expectations checks
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │      ingest      │  loads into PostgreSQL
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │ build_features   │  Pandas/NumPy feature engineering
                └────────┬─────────┘
                         ▼
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌───────────────────┐       ┌───────────────────┐
│  train_sklearn.py  │       │  train_torch.py    │   both log to MLflow
└─────────┬──────────┘       └──────────┬─────────┘
          └──────────────┬──────────────┘
                         ▼
                ┌─────────────────┐
                │   models/*       │  saved model artifacts
                └────────┬─────────┘
                         ▼
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌───────────────────┐       ┌───────────────────┐
│  FastAPI (api/)    │       │  Streamlit (app/)  │
│  /predict endpoint │       │  dashboard + form  │
└────────────────────┘       └────────────────────┘
```

## Orchestration

`src/churn/pipelines/airflow_dag.py` wires the steps above (generate →
validate → ingest → features → train) into a daily Airflow DAG, so the
whole pipeline can be scheduled and monitored rather than run by hand.

## Why these tools

- **Great Expectations** sits right after data generation/collection so bad
  data never reaches the database or a model.
- **PostgreSQL** is the system of record for customer data — useful once
  data comes from a real source rather than a CSV.
- **MLflow** tracks every training run's parameters, metrics, and model
  artifacts, so runs are comparable and reproducible.
- **FastAPI** exposes the model for programmatic access (e.g. from another
  service); **Streamlit** gives humans an interactive view.
- **Airflow** automates the whole thing on a schedule.
- **pytest + Ruff + Black + GitHub Actions** keep the codebase correct,
  consistent, and continuously verified.
