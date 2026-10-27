"""Run: python test_api.py  (needs models/clv.joblib from python -m clv.train)"""
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
good = {"recency_days": 5, "tenure_days": 300, "frequency": 40, "monetary": 50000,
        "revenue_last_6m": 25000, "orders_last_90d": 10, "n_products": 200, "is_uk": True}
lapsed = good | {"recency_days": 300, "frequency": 1, "monetary": 50, "revenue_last_6m": 0,
                 "orders_last_90d": 0, "n_products": 2}

assert client.get("/health").json()["status"] == "ok"

hi, lo = client.post("/predict", json=good).json(), client.post("/predict", json=lapsed).json()
assert hi["segment"] == "high" and lo["segment"] == "low", (hi, lo)
assert hi["predicted_clv_6m"] > lo["predicted_clv_6m"] >= 0

for bad in [good | {"frequency": 0}, good | {"recency_days": 400},
            good | {"revenue_last_6m": 60000}, good | {"recency_days": 310}]:
    assert client.post("/predict", json=bad).status_code == 422, bad

print("ok", hi, lo)
