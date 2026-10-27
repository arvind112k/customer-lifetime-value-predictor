"""Customer features as of a cutoff date, plus the 6-month future revenue target.

Features use only transactions in [cutoff - 12 months, cutoff).
Target is revenue in [cutoff, cutoff + 6 months). Customers who don't buy again get 0.

Run: python -m clv.features
"""
import pandas as pd

LOOKBACK = pd.DateOffset(months=12)
HORIZON = pd.DateOffset(months=6)
TRAIN_CUTOFF = pd.Timestamp("2010-12-01")
TEST_CUTOFF = pd.Timestamp("2011-06-01")

FEATURES = ["recency_days", "tenure_days", "frequency", "monetary", "avg_order_value",
            "revenue_last_6m", "orders_last_90d", "n_products", "is_uk"]


def build(tx, cutoff):
    cutoff = pd.Timestamp(cutoff)
    past = tx[(tx.date >= cutoff - LOOKBACK) & (tx.date < cutoff)]
    future = tx[(tx.date >= cutoff) & (tx.date < cutoff + HORIZON)]

    days_ago = (cutoff - past.date).dt.days
    g = past.assign(days_ago=days_ago).groupby("customer_id")
    df = pd.DataFrame({
        "recency_days": g.days_ago.min(),
        "tenure_days": g.days_ago.max(),
        "frequency": g.invoice.nunique(),
        "monetary": g.revenue.sum(),
        "n_products": g.stock_code.nunique(),
        "is_uk": g.country.agg(lambda c: c.mode()[0] == "United Kingdom").astype(int),
    })
    df["avg_order_value"] = df.monetary / df.frequency
    recent = past[days_ago < 183]  # ~6 months, also the naive baseline's prediction
    df["revenue_last_6m"] = recent.groupby("customer_id").revenue.sum()
    df["orders_last_90d"] = past[days_ago < 90].groupby("customer_id").invoice.nunique()
    df["target"] = future.groupby("customer_id").revenue.sum()
    return df.fillna(0)[FEATURES + ["target"]]


def main():
    tx = pd.read_parquet("data/transactions.parquet")
    for name, cutoff in [("train", TRAIN_CUTOFF), ("test", TEST_CUTOFF)]:
        df = build(tx, cutoff)
        df.to_parquet(f"data/{name}.parquet")
        print(f"{name}: cutoff {cutoff.date()}, {len(df):,} customers, "
              f"{(df.target == 0).mean():.0%} spend nothing next 6m, "
              f"mean target £{df.target.mean():,.0f}")


if __name__ == "__main__":
    main()
