# Setup

## Prerequisites

- Python 3.11
- Conda (recommended) or `venv`
- PostgreSQL (local install or Docker)
- Git

## Install

```bash
conda env create -f environment.yml
conda activate churn-predictor
pip install -e .
cp .env.example .env
```

Edit `.env` with your PostgreSQL credentials.

## Run the pipeline

```bash
make data       # generate synthetic customer data
make validate   # check data quality with Great Expectations
make ingest     # load data into PostgreSQL
make features   # build the model-ready feature table
make train      # train both models, tracked in MLflow
```

## Run the apps

```bash
make api         # FastAPI on http://localhost:8000/docs
make dashboard   # Streamlit on http://localhost:8501
make mlflow-ui   # MLflow UI on http://localhost:5000
```

## Run tests and checks

```bash
make test
make lint
```

For a fully guided, no-assumptions walkthrough, see `EXECUTION_GUIDE.docx`
in the project root.
