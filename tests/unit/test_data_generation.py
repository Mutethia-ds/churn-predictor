from __future__ import annotations

from churn.data.generate_data import generate_customers


def test_generate_customers_row_count():
    df = generate_customers(n_samples=100, seed=42)
    assert len(df) == 100


def test_generate_customers_columns():
    df = generate_customers(n_samples=10, seed=42)
    expected = {
        "customer_id",
        "tenure_months",
        "contract_type",
        "internet_service",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "support_calls",
        "churn",
    }
    assert expected.issubset(set(df.columns))


def test_generate_customers_is_deterministic_given_seed():
    df1 = generate_customers(n_samples=50, seed=7)
    df2 = generate_customers(n_samples=50, seed=7)
    assert df1.equals(df2)


def test_churn_is_binary():
    df = generate_customers(n_samples=200, seed=1)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_customer_ids_are_unique():
    df = generate_customers(n_samples=500, seed=3)
    assert df["customer_id"].is_unique
