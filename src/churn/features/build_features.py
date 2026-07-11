"""Feature engineering: turn raw customer data into a model-ready table."""

from __future__ import annotations

import pandas as pd

from churn.config import PROJECT_ROOT, get_settings


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)
    df["high_support_usage"] = (df["support_calls"] >= 3).astype(int)
    return df


def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def build_feature_table(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = add_derived_features(df)
    categorical_cols = settings["features"]["categorical"]
    df = one_hot_encode(df, categorical_cols)
    return df


def main() -> None:
    settings = get_settings()
    raw_path = PROJECT_ROOT / settings["data"]["raw_path"]
    processed_path = PROJECT_ROOT / settings["data"]["processed_path"]

    df = pd.read_csv(raw_path)
    feature_df = build_feature_table(df, settings)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(processed_path, index=False)
    print(f"Wrote {len(feature_df)} rows x {feature_df.shape[1]} cols to {processed_path}")


if __name__ == "__main__":
    main()
