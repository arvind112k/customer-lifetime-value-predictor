"""Streamlit front end. Needs the API running (uvicorn api:app).

Run: streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
ACTIONS = {
    "high": "Top 20% — put them on the retention campaign.",
    "medium": "Next 30% — nurture with low-cost email.",
    "low": "Bottom 50% — don't spend retention budget here.",
}

st.set_page_config(page_title="CLV Predictor", page_icon="💷")
st.title("Customer Lifetime Value Predictor")
st.caption("Predicts a customer's revenue over the next 6 months from their last 12 months of orders.")

predict_tab, results_tab = st.tabs(["Predict", "Model results"])

with predict_tab:
    with st.form("customer"):
        a, b = st.columns(2)
        recency = a.number_input("Days since last purchase", 0, 365, 20)
        tenure = b.number_input("Days since first purchase (max 365)", 0, 365, 300)
        frequency = a.number_input("Orders in last 12 months", 1, 10_000, 8)
        orders_90 = b.number_input("Orders in last 90 days", 0, 10_000, 2)
        monetary = a.number_input("Revenue last 12 months (£)", 0.01, 1e7, 3200.0)
        rev_6m = b.number_input("Revenue last 6 months (£)", 0.0, 1e7, 1500.0)
        n_products = a.number_input("Distinct products bought", 1, 10_000, 60)
        is_uk = b.checkbox("UK customer", True)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"recency_days": recency, "tenure_days": tenure, "frequency": frequency,
                   "monetary": monetary, "revenue_last_6m": rev_6m, "orders_last_90d": orders_90,
                   "n_products": n_products, "is_uk": is_uk}
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=10)
        except requests.ConnectionError:
            st.error(f"Can't reach the API at {API}. Start it with `uvicorn api:app`.")
        else:
            if r.ok:
                p = r.json()
                st.metric("Predicted revenue, next 6 months", f"£{p['predicted_clv_6m']:,.0f}")
                st.info(f"Segment: **{p['segment']}**. {ACTIONS[p['segment']]}")
            else:
                for err in r.json()["detail"]:
                    st.error(err["msg"].removeprefix("Value error, "))

with results_tab:
    st.subheader("Out-of-time test: features as of Jun 2011, revenue Jun–Nov 2011")
    metrics = pd.read_json("models/metrics.json", orient="index")
    st.dataframe(metrics.style.format({"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "top20_capture": "{:.1%}",
                                       "top10_lift": "{:.2f}×", "total_bias": "{:+.0%}"}))
    st.markdown("**top20_capture**: if marketing targets the model's top 20% of customers, "
                "this share of next period's revenue is inside that group. Higher is better.")

    preds = "data/test_predictions.parquet"
    if os.path.exists(preds):
        df = pd.read_parquet(preds)
        y = df.target.values
        # Gains curve, sampled at 1% steps
        share = np.linspace(0, 1, 101)
        curves = {"perfect": y}
        curves |= {c.removeprefix("pred_"): df[c].values for c in df if c.startswith("pred_")}
        gains = pd.DataFrame({n: np.interp(share, np.arange(1, len(y) + 1) / len(y),
                                           y[np.argsort(-p)].cumsum() / y.sum())
                              for n, p in curves.items()}, index=share)
        st.subheader("Gains curve")
        st.line_chart(gains, x_label="share of customers targeted", y_label="share of revenue captured")
