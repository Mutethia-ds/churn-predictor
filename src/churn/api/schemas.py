"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0, le=100, examples=[12])
    contract_type: str = Field(examples=["month-to-month"])
    internet_service: str = Field(examples=["fiber"])
    payment_method: str = Field(examples=["electronic_check"])
    monthly_charges: float = Field(ge=0, examples=[74.5])
    total_charges: float = Field(ge=0, examples=[894.0])
    support_calls: int = Field(ge=0, examples=[2])


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
