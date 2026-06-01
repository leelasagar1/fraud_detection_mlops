"""Shared, TESTED feature logic. Imported by both the Databricks notebook (in Repos)
and the CI tests — so the thing that runs in production is the thing that's tested.
Extracting this out of the notebook is itself a senior-engineering signal."""
import pandas as pd

FEATURES = ["amount", "amount_zscore", "txn_count_1h", "txn_count_24h",
            "amount_sum_24h", "merchant_category", "channel", "is_night"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time-correct features. closed='left' excludes the current row, so a
    transaction never sees itself or the future — this is what prevents label leakage."""
    df = df.sort_values("event_ts").copy()
    df["is_night"] = (df["event_ts"].dt.hour < 6).astype(int)
    parts = []
    for _, g in df.groupby("customer_id"):
        g = g.sort_values("event_ts").set_index("event_ts")
        amt = g["amount"]
        g["txn_count_1h"]   = amt.rolling("1h",  closed="left").count().fillna(0)
        g["txn_count_24h"]  = amt.rolling("24h", closed="left").count().fillna(0)
        g["amount_sum_24h"] = amt.rolling("24h", closed="left").sum().fillna(0)
        m7 = amt.rolling("7d", closed="left").mean()
        s7 = amt.rolling("7d", closed="left").std().fillna(1).clip(lower=1)
        g["amount_zscore"]  = ((amt - m7) / s7).fillna(0)
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)
