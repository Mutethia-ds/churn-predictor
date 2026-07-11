"""FastAPI app that serves churn predictions.

Run with:
    uvicorn churn.api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from churn.api.schemas import CustomerInput, HealthResponse, PredictionOutput
from churn.models.predict import load_model, predict_one

app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predicts the probability that a customer will churn.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput) -> PredictionOutput:
    try:
        result = predict_one(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionOutput(**result)


@app.get("/")
def root() -> dict:
    return {"message": "Customer Churn Predictor API. See /docs for usage."}
