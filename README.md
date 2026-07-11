# Customer Churn Predictor

An end-to-end, portfolio-ready data science project. It takes a customer
dataset all the way from raw data to a served model, with proper testing,
CI/CD, and documentation.

## What this project does

Predicts whether a telecom customer will churn (cancel their subscription)
using their account and usage data. It includes:

- **Data validation** with Great Expectations
- **A PostgreSQL database** for storing customer records
- **Feature engineering** with Pandas / NumPy
- **Two trained models**: a scikit-learn gradient boosting model and a
  PyTorch neural network, both tracked with MLflow
- **An orchestrated pipeline** with Apache Airflow
- **A REST API** (FastAPI) that serves predictions
- **A dashboard** (Streamlit) for exploring data and predictions
- **Tests** (pytest, unit + integration)
- **Linting/formatting** (Ruff + Black)
- **CI/CD** (GitHub Actions) that lints, tests, and validates data on every push
- **Docs** (MkDocs)

## Tech stack

| Layer | Tool |
|---|---|
| Editor | VS Code |
| Version control | Git + GitHub |
| Environment | Conda or venv |
| Data | Pandas, NumPy, PostgreSQL |
| ML | scikit-learn, PyTorch |
| Viz | Matplotlib, Plotly |
| Apps | Streamlit, FastAPI |
| Experiment tracking | MLflow |
| Data quality | Great Expectations |
| Orchestration | Apache Airflow |
| Testing | pytest |
| Lint/format | Ruff + Black |
| Config | YAML + .env |
| CI/CD | GitHub Actions |
| Docs | Markdown + MkDocs |

## Quick start

See **`EXECUTION_GUIDE.docx`** for a plain-English, step-by-step walkthrough
that assumes zero prior setup. The short version:

```bash
conda env create -f environment.yml
conda activate churn-predictor
cp .env.example .env
make data          # generate + validate synthetic data
make train         # train models, log to MLflow
make api           # run the FastAPI server
make dashboard     # run the Streamlit app
make test          # run pytest
```

## Project layout

```
churn-predictor/
├── .github/workflows/ci.yml     # GitHub Actions pipeline
├── app/dashboard.py              # Streamlit dashboard
├── config/config.yaml            # non-secret configuration
├── docs/                         # MkDocs source
├── great_expectations/           # data validation suite
├── notebooks/                    # exploratory notebooks
├── src/churn/
│   ├── data/                     # data generation, ingestion, validation
│   ├── db/                       # SQLAlchemy models + session
│   ├── features/                 # feature engineering
│   ├── models/                   # training + inference (sklearn & PyTorch)
│   ├── api/                      # FastAPI app
│   └── pipelines/                # Airflow DAG
├── tests/
│   ├── unit/
│   └── integration/
├── environment.yml / requirements.txt
├── pyproject.toml                # Ruff + Black + pytest config
├── mkdocs.yml
├── Makefile
└── .env.example
```

## License

MIT — use this as a portfolio template freely.
