from __future__ import annotations

from churn.features.build_features import add_derived_features, one_hot_encode


def test_add_derived_features_creates_expected_columns(sample_raw_df):
    result = add_derived_features(sample_raw_df)

    assert "avg_monthly_spend" in result.columns
    assert "is_new_customer" in result.columns
    assert "high_support_usage" in result.columns


def test_add_derived_features_handles_zero_tenure(sample_raw_df):
    # tenure_months == 1 in the fixture's first row; ensure no division-by-zero
    # crash even for a genuinely zero-tenure row.
    sample_raw_df.loc[0, "tenure_months"] = 0
    result = add_derived_features(sample_raw_df)
    assert result.loc[0, "avg_monthly_spend"] == sample_raw_df.loc[0, "total_charges"]


def test_is_new_customer_flag(sample_raw_df):
    result = add_derived_features(sample_raw_df)
    assert result.loc[0, "is_new_customer"] == 1  # tenure_months = 1
    assert result.loc[2, "is_new_customer"] == 0  # tenure_months = 60


def test_high_support_usage_flag(sample_raw_df):
    result = add_derived_features(sample_raw_df)
    assert result.loc[0, "high_support_usage"] == 1  # 4 calls
    assert result.loc[2, "high_support_usage"] == 0  # 0 calls


def test_one_hot_encode_drops_original_categorical_columns(sample_raw_df):
    result = one_hot_encode(sample_raw_df, ["contract_type", "internet_service"])
    assert "contract_type" not in result.columns
    assert "internet_service" not in result.columns
    assert any(col.startswith("contract_type_") for col in result.columns)
