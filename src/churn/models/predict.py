"""Load the trained sklearn model and run predictions on new customers."""

from __future__ import annotations

from functools import lru_cache

import mlflow.sklearn
import pandas as pd

from churn.config import PROJECT_ROOT, get_settings
from churn.features.build_features import build_feature_table


@lru_cache
def load_model():
    model_path = PROJECT_ROOT / "models" / "sklearn_model"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run `make train` first.")
    return mlflow.sklearn.load_model(str(model_path))


def predict_one(customer: dict) -> dict:
    """customer: dict of raw feature values (same shape as one row of the
    raw customers CSV, minus customer_id/churn)."""
    settings = get_settings()
    model = load_model()

    df = pd.DataFrame([customer])
    feature_df = build_feature_table(df, settings)
    feature_df = feature_df.drop(
        columns=[c for c in ["churn", "customer_id"] if c in feature_df.columns]
    )

    # Align columns with what the model was trained on, filling missing
    # one-hot columns (e.g. a category not present in this single row) with 0.
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[expected_cols]

    proba = model.predict_proba(feature_df)[0, 1]
    pred = int(proba >= 0.5)
    return {"churn_probability": float(proba), "churn_prediction": pred}
