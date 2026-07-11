from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST-000001", "CUST-000002", "CUST-000003"],
            "tenure_months": [1, 24, 60],
            "contract_type": ["month-to-month", "one-year", "two-year"],
            "internet_service": ["fiber", "dsl", "none"],
            "payment_method": [
                "electronic_check",
                "credit_card",
                "bank_transfer",
            ],
            "monthly_charges": [90.0, 55.0, 40.0],
            "total_charges": [90.0, 1320.0, 2400.0],
            "support_calls": [4, 1, 0],
            "churn": [1, 0, 0],
        }
    )
