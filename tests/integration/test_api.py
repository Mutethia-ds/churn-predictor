"""Integration tests for the FastAPI app.

These use FastAPI's TestClient, so no live server is needed - but a trained
model must exist at models/sklearn_model (run `make train` first), or the
prediction test will be skipped.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.config import PROJECT_ROOT

client = TestClient(app)

MODEL_EXISTS = (PROJECT_ROOT / "models" / "sklearn_model").exists()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet; run `make train`")
def test_predict_endpoint_returns_valid_probability():
    payload = {
        "tenure_months": 3,
        "contract_type": "month-to-month",
        "internet_service": "fiber",
        "payment_method": "electronic_check",
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "support_calls": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_endpoint_rejects_invalid_input():
    payload = {"tenure_months": -5}  # invalid, and missing required fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
