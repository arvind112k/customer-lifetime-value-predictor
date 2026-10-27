"""Leakage check: changing anything on/after the cutoff must not change features.

Run: python test_features.py
"""
import pandas as pd

from clv.features import FEATURES, build

CUTOFF = pd.Timestamp("2011-01-01")


def tx(rows):
    return pd.DataFrame(rows, columns=["customer_id", "invoice", "date", "stock_code", "revenue", "country"]) \
        .assign(date=lambda d: pd.to_datetime(d.date))


base = tx([
    (1, "a", "2010-12-20", "X", 100.0, "United Kingdom"),  # 12 days before cutoff
    (1, "b", "2010-03-01", "Y", 50.0, "United Kingdom"),
    (2, "c", "2010-10-01", "X", 30.0, "France"),
    (1, "d", "2011-02-01", "X", 70.0, "United Kingdom"),   # future -> target only
    (3, "e", "2009-06-01", "X", 999.0, "France"),          # outside 12m lookback
])
f = build(base, CUTOFF)

# Hand-checked values
assert list(f.index) == [1, 2], f.index  # customer 3 excluded: no activity in lookback
assert f.loc[1, "recency_days"] == 12
assert f.loc[1, "frequency"] == 2 and f.loc[1, "monetary"] == 150
assert f.loc[1, "revenue_last_6m"] == 100  # March purchase is >6m old
assert f.loc[1, "target"] == 70 and f.loc[2, "target"] == 0
assert f.loc[1, "is_uk"] == 1 and f.loc[2, "is_uk"] == 0

# Leakage: add a big purchase exactly at the cutoff and later -> features identical, target changes
leaky = pd.concat([base, tx([
    (2, "f", "2011-01-01", "Z", 5000.0, "France"),
    (1, "g", "2011-05-01", "Z", 5000.0, "United Kingdom"),
])])
g = build(leaky, CUTOFF)
pd.testing.assert_frame_equal(f[FEATURES], g[FEATURES])
assert g.loc[2, "target"] == 5000 and g.loc[1, "target"] == 5070

print("ok")
