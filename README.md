# Customer Lifetime Value Predictor

Predicts how much revenue each customer will bring in over the **next 6 months**, so a marketing team can decide **who gets retention budget**. Built on ~780k real transactions from a UK online retailer, compared honestly against a naive baseline, with error analysis and a FastAPI + Streamlit deployment.

## The business question

Retention campaigns cost money per customer. If you can only afford to target **20% of customers**, which 20%? The right metric isn't MAE — it's **how much of next period's revenue sits inside the group you picked**:

- **Top-20% capture** — share of actual future revenue held by the model's top 20% of customers.
- **Top-10% lift** — how many times better than random the model's top 10% is.
- **Total bias** — does the model's total forecast match what actually happened? (Matters if finance uses it.)

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1.07M invoice lines, Dec 2009 – Dec 2011. After removing cancellations, missing customer IDs and non-positive quantities/prices: **779,425 rows, 5,878 customers**.

## Approach

**Out-of-time split, no leakage.** Features use only the 12 months before a cutoff; the target is revenue in the 6 months after it. Customers who don't come back have a target of £0.

| | Features from | Target window | Customers |
|---|---|---|---|
| Train | Dec 2009 – Nov 2010 | Dec 2010 – May 2011 | 4,266 |
| Test | Jun 2010 – May 2011 | Jun 2011 – Nov 2011 | 4,338 |

`test_features.py` proves no leakage: adding purchases on or after the cutoff must leave every feature unchanged.

**Features:** recency, tenure, frequency, total and average order value, revenue in last 6 months, orders in last 90 days, distinct products, UK flag.

**Models:** baseline ("next 6 months = last 6 months"), Ridge on log scale, Random Forest, LightGBM (Tweedie objective, which suits a zero-heavy, long-tailed target).

## Results (test set)

| Model | MAE | RMSE | Top-20% capture | Top-10% lift | Total bias |
|---|---|---|---|---|---|
| Baseline (last 6m) | **£643** | **£3,233** | 72.4% | 5.99× | **−21%** |
| Ridge (log) | £1,015 | £11,265 | 72.4% | 5.90× | −41% |
| **Random Forest** | £697 | £4,260 | **73.7%** | **6.15×** | −42% |
| LightGBM Tweedie | £670 | £3,817 | 73.5% | 6.15× | −49% |

Random Forest is chosen on top-20% capture: picking who to target is a ranking problem.

**The honest headline:** the model ranks customers better than the baseline (+£54k of future revenue inside a top-20% campaign), but the baseline wins on MAE, and no model should be used as a revenue forecast without correction.

## Error analysis — what's actually going on

Full notebook: [`clv/analysis.ipynb`](clv/analysis.ipynb).

1. **Zero inflation.** 43% of customers spend £0 next period, and the top 1% hold 36% of future revenue. Average-error metrics are dominated by a handful of customers.
2. **Seasonality causes the −42% bias.** Train labels cover Dec–May; the test window covers Jun–Nov, which includes the Sep–Nov wholesale peak (~1.7× spring revenue in both years). The model learned to shrink history into an off-season label, then got asked about peak season.
3. **Trees can't predict whales.** Random Forest predictions cap at ~£23.7k; the most under-predicted customers spent £74k–£184k. Top accounts need a key-account manager, not a model.
4. **The baseline can't see reactivation.** Customers silent for 180+ days get a baseline prediction of £0 but spend £230 on average. The model catches part of this, and that's where its ranking gain comes from.
5. **Costly false positives are one-off bulk buyers** (e.g. £77k in 2 orders, then nothing). A simple rule could flag them before retention money is spent.

**Recommendation:** use the model to rank customers for retention spend, handle the top ~1% manually, and apply a seasonal correction before quoting totals. The gap to a perfect ranking is still £418k, so better features (product mix, order-interval trends) have room to help.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m clv.data       # download + clean (~1 min)
python -m clv.features   # build train/test feature tables
python -m clv.train      # compare models, save best to models/

uvicorn api:app          # API on http://localhost:8000  (docs at /docs)
streamlit run app.py     # UI on http://localhost:8501  (separate terminal)
```

Checks: `python test_features.py` (leakage) and `python test_api.py` (API validation).

### API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"recency_days":20,"tenure_days":300,"frequency":8,"monetary":3200,"revenue_last_6m":1500,"orders_last_90d":2,"n_products":60,"is_uk":true}'
# {"predicted_clv_6m":925.85,"segment":"high","model":"random_forest"}
```

Segments: `high` = top 20% of predicted CLV (retention target), `medium` = next 30%, `low` = bottom 50%. Inconsistent inputs (e.g. last purchase before first purchase) are rejected with a 422.

## Layout

```
clv/data.py          download + clean
clv/features.py      leak-free features and target at a cutoff
clv/train.py         baseline vs models, business metrics, saves best model
clv/analysis.ipynb   error analysis
api.py               FastAPI service
app.py               Streamlit front end
test_features.py     leakage test
test_api.py          API test
```

Note: `matplotlib` is pinned to 3.9.4 because newer Windows wheels are blocked by Smart App Control on some machines.
