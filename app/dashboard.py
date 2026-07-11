"""Streamlit dashboard: explore the churn dataset and get live predictions.

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import PROJECT_ROOT, get_settings  # noqa: E402
from churn.models.predict import predict_one  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Predictor")

settings = get_settings()
raw_path = PROJECT_ROOT / settings["data"]["raw_path"]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(raw_path)


tab_explore, tab_predict = st.tabs(["Explore data", "Predict churn"])

with tab_explore:
    if not raw_path.exists():
        st.warning("No data found yet. Run `make data` first.")
    else:
        df = load_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Customers", len(df))
        col2.metric("Churn rate", f"{df['churn'].mean():.1%}")
        col3.metric("Avg. monthly charges", f"${df['monthly_charges'].mean():.2f}")

        st.plotly_chart(
            px.histogram(
                df,
                x="contract_type",
                color="churn",
                barmode="group",
                title="Churn by contract type",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.box(
                df,
                x="churn",
                y="monthly_charges",
                color="churn",
                title="Monthly charges vs churn",
            ),
            use_container_width=True,
        )
        st.dataframe(df.head(100))

with tab_predict:
    st.subheader("Score a single customer")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure_months = st.slider("Tenure (months)", 0, 72, 12)
            contract_type = st.selectbox(
                "Contract type", ["month-to-month", "one-year", "two-year"]
            )
            internet_service = st.selectbox("Internet service", ["dsl", "fiber", "none"])
        with c2:
            payment_method = st.selectbox(
                "Payment method",
                ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
            )
            monthly_charges = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.0)
            total_charges = st.number_input("Total charges ($)", 0.0, 20000.0, 840.0)
            support_calls = st.slider("Support calls", 0, 20, 1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        customer = {
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
        }
        try:
            result = predict_one(customer)
            st.metric("Churn probability", f"{result['churn_probability']:.1%}")
            if result["churn_prediction"] == 1:
                st.error("Prediction: likely to churn")
            else:
                st.success("Prediction: likely to stay")
        except FileNotFoundError as e:
            st.error(str(e))
