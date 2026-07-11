"""Generate a synthetic telecom customer churn dataset.

Using a synthetic generator (rather than an external download) keeps this
project fully reproducible and network-independent, while still exercising
the exact same pipeline you'd use on a real dataset like the IBM Telco
Churn dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import PROJECT_ROOT, get_secrets, get_settings


def generate_customers(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, size=n_samples)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(["dsl", "fiber", "none"], size=n_samples, p=[0.35, 0.45, 0.20])
    payment_method = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        size=n_samples,
    )
    monthly_charges = np.round(rng.normal(65, 25, size=n_samples).clip(15, 150), 2)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 50, n_samples), 2)
    total_charges = total_charges.clip(min=0)
    support_calls = rng.poisson(1.5, size=n_samples)

    # Build churn probability from a hand-crafted logistic function so the
    # model has real signal to learn.
    logit = (
        -1.5
        + 0.03 * (24 - tenure_months)
        + 0.015 * (monthly_charges - 65)
        + 0.35 * support_calls
        + np.where(contract_type == "month-to-month", 1.1, 0.0)
        + np.where(payment_method == "electronic_check", 0.4, 0.0)
        - np.where(contract_type == "two-year", 0.8, 0.0)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_samples)],
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "churn": churn,
        }
    )
    return df


def main() -> Path:
    settings = get_settings()
    secrets = get_secrets()

    df = generate_customers(n_samples=settings["data"]["n_samples"], seed=secrets.random_seed)

    out_path = PROJECT_ROOT / settings["data"]["raw_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
