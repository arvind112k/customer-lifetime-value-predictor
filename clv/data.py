"""Download UCI Online Retail II and save a clean transactions table.

Run: python -m clv.data
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA = Path("data")
RAW = DATA / "online_retail_II.xlsx"
OUT = DATA / "transactions.parquet"


def download():
    if RAW.exists():
        return
    DATA.mkdir(exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        xlsx = next(n for n in z.namelist() if n.endswith(".xlsx"))
        RAW.write_bytes(z.read(xlsx))


def clean(df):
    df = df.rename(columns={"Customer ID": "customer_id", "InvoiceDate": "date",
                            "Price": "price", "Quantity": "qty", "Invoice": "invoice",
                            "StockCode": "stock_code", "Country": "country"})
    df = df.dropna(subset=["customer_id"])
    df = df[~df["invoice"].astype(str).str.startswith("C")]  # cancellations
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    df = df.drop_duplicates()
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["revenue"] = df["qty"] * df["price"]
    return df[["customer_id", "invoice", "date", "stock_code", "qty", "price", "revenue", "country"]]


def main():
    download()
    # File has two sheets: 2009-2010 and 2010-2011
    df = pd.concat(pd.read_excel(RAW, sheet_name=None).values(), ignore_index=True)
    tx = clean(df)
    tx.to_parquet(OUT, index=False)
    print(f"{len(df):,} raw rows -> {len(tx):,} clean rows, "
          f"{tx.customer_id.nunique():,} customers, {tx.date.min()} to {tx.date.max()}")


if __name__ == "__main__":
    main()
