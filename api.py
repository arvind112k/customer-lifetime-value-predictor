"""CLV prediction API.

Run: uvicorn api:app --reload
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

bundle = joblib.load("models/clv.joblib")
app = FastAPI(title="Customer Lifetime Value Predictor")


class Customer(BaseModel):
    """A customer's last 12 months of activity, as of today."""
    recency_days: int = Field(ge=0, le=365, description="Days since last purchase")
    tenure_days: int = Field(ge=0, le=365, description="Days since first purchase in the last 12 months")
    frequency: int = Field(ge=1, description="Number of orders in the last 12 months")
    monetary: float = Field(gt=0, description="Total revenue (£) in the last 12 months")
    revenue_last_6m: float = Field(ge=0, description="Revenue (£) in the last 6 months")
    orders_last_90d: int = Field(ge=0, description="Orders in the last 90 days")
    n_products: int = Field(ge=1, description="Distinct products bought")
    is_uk: bool

    @model_validator(mode="after")
    def consistent(self):
        if self.recency_days > self.tenure_days:
            raise ValueError("recency_days cannot exceed tenure_days")
        if self.revenue_last_6m > self.monetary:
            raise ValueError("revenue_last_6m cannot exceed monetary")
        if self.orders_last_90d > self.frequency:
            raise ValueError("orders_last_90d cannot exceed frequency")
        return self


class Prediction(BaseModel):
    predicted_clv_6m: float
    segment: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["name"]}


@app.post("/predict", response_model=Prediction)
def predict(c: Customer):
    row = c.model_dump() | {"avg_order_value": c.monetary / c.frequency, "is_uk": int(c.is_uk)}
    clv = max(0.0, float(bundle["model"].predict(pd.DataFrame([row])[bundle["features"]])[0]))
    tiers = bundle["tiers"]
    segment = "high" if clv >= tiers["high"] else "medium" if clv >= tiers["medium"] else "low"
    return Prediction(predicted_clv_6m=round(clv, 2), segment=segment, model=bundle["name"])
