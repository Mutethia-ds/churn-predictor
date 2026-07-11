"""Validate the raw dataset with Great Expectations before it enters the
pipeline. Fails loudly (non-zero exit code) if any expectation is violated,
so CI and Airflow both stop on bad data.
"""

from __future__ import annotations

import sys

import great_expectations as gx
import pandas as pd

from churn.config import PROJECT_ROOT, get_settings


def build_expectation_suite(validator: gx.validator.validator.Validator) -> None:
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_unique("customer_id")

    validator.expect_column_values_to_be_between("tenure_months", min_value=0, max_value=100)
    validator.expect_column_values_to_be_between("monthly_charges", min_value=0, max_value=1000)
    validator.expect_column_values_to_be_between("total_charges", min_value=0, max_value=100000)
    validator.expect_column_values_to_be_between("support_calls", min_value=0, max_value=50)

    validator.expect_column_values_to_be_in_set(
        "contract_type", ["month-to-month", "one-year", "two-year"]
    )
    validator.expect_column_values_to_be_in_set("internet_service", ["dsl", "fiber", "none"])
    validator.expect_column_values_to_be_in_set(
        "payment_method",
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
    )
    validator.expect_column_values_to_be_in_set("churn", [0, 1])

    validator.expect_table_row_count_to_be_between(min_value=1)


def validate(csv_path=None) -> bool:
    settings = get_settings()
    csv_path = csv_path or (PROJECT_ROOT / settings["data"]["raw_path"])

    df = pd.read_csv(csv_path)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("runtime_pandas")
    asset = data_source.add_dataframe_asset(name="customers")
    batch_definition = asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    validator = context.get_validator(batch=batch)
    build_expectation_suite(validator)

    result = validator.validate()
    print(f"Great Expectations: success={result.success}")
    if not result.success:
        for res in result.results:
            if not res.success:
                print(f"  FAILED: {res.expectation_config.type}")
    return bool(result.success)


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
