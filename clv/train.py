"""Compare a naive baseline against three models on the out-of-time test set.

Run: python -m clv.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from clv.features import FEATURES

MODELS = Path("models")


def capture(y, pred, top=0.2):
    """Share of actual future revenue held by the customers the model ranks in its top `top` fraction."""
    k = int(len(y) * top)
    return y[np.argsort(-pred)[:k]].sum() / y.sum()


def evaluate(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "top20_capture": capture(y, pred),
        "top10_lift": capture(y, pred, 0.1) / 0.1,
        "total_bias": pred.sum() / y.sum() - 1,  # +0.1 = forecasts 10% more revenue than happened
    }


def models():
    log = FunctionTransformer(np.log1p)
    return {
        # Log features + log target: linear in "orders of magnitude", then back to £
        "ridge_log": TransformedTargetRegressor(
            make_pipeline(log, StandardScaler(), Ridge()), func=np.log1p, inverse_func=np.expm1),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0),
        # Tweedie suits revenue: a point mass at 0 plus a long right tail
        "lgbm_tweedie": LGBMRegressor(objective="tweedie", n_estimators=300, learning_rate=0.03,
                                      num_leaves=15, min_child_samples=30, random_state=0, verbose=-1),
    }


def main():
    assert capture(np.array([0, 0, 10, 90, 0.0]), np.array([1, 2, 3, 4, 5.0]), 0.4) == 0.9

    train, test = pd.read_parquet("data/train.parquet"), pd.read_parquet("data/test.parquet")
    X, y, Xt, yt = train[FEATURES], train.target.values, test[FEATURES], test.target.values

    preds = {"baseline_last_6m": Xt.revenue_last_6m.values}
    fitted = {}
    for name, m in models().items():
        fitted[name] = m.fit(X, y)
        preds[name] = np.clip(m.predict(Xt), 0, None)

    results = pd.DataFrame({n: evaluate(yt, p) for n, p in preds.items()}).T
    print(results.round(3).to_string())

    # Pick on the business metric: who to spend retention budget on is a ranking problem
    best = results.drop("baseline_last_6m").top20_capture.idxmax()
    print(f"\nbest: {best}")

    MODELS.mkdir(exist_ok=True)
    # Segment cut-points from training predictions: top 20% = retention target, next 30% = nurture
    train_pred = fitted[best].predict(X)
    tiers = {"high": float(np.quantile(train_pred, 0.8)), "medium": float(np.quantile(train_pred, 0.5))}
    joblib.dump({"model": fitted[best], "features": FEATURES, "name": best, "tiers": tiers}, MODELS / "clv.joblib")
    (MODELS / "metrics.json").write_text(results.to_json(orient="index", indent=2))
    test.assign(**{f"pred_{n}": p for n, p in preds.items()}).to_parquet("data/test_predictions.parquet")


if __name__ == "__main__":
    main()
