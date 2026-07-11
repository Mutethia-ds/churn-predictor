.PHONY: install data validate ingest features train train-sklearn train-torch \
        api dashboard test lint format docs mlflow-ui clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python -m churn.data.generate_data

validate:
	python -m churn.data.validate_data

ingest:
	python -m churn.data.ingest

features:
	python -m churn.features.build_features

train: train-sklearn train-torch

train-sklearn:
	python -m churn.models.train_sklearn

train-torch:
	python -m churn.models.train_torch

api:
	uvicorn churn.api.main:app --reload --port 8000

dashboard:
	streamlit run app/dashboard.py

test:
	pytest

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

docs:
	mkdocs serve

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov site
