"""Tests that pin the property that matters most: no future leakage."""
import pandas as pd
from src.features import build_features, FEATURES


def _df():
    return pd.DataFrame({
        "transaction_id": range(3),
        "customer_id": [1, 1, 1],
        "event_ts": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 11:00", "2025-01-02 09:00"]),
        "amount": [100.0, 50.0, 25.0],
        "merchant_category": ["retail", "grocery", "fuel"],
        "channel": ["online", "in_person", "in_person"],
        "is_fraud": [0, 0, 0],
    })


def test_no_leakage_on_first_row():
    out = build_features(_df()).sort_values("event_ts").reset_index(drop=True)
    # First-ever transaction has no prior history.
    assert out.loc[0, "txn_count_24h"] == 0
    assert out.loc[0, "amount_sum_24h"] == 0


def test_counts_only_past_within_window():
    out = build_features(_df()).sort_values("event_ts").reset_index(drop=True)
    assert out.loc[1, "txn_count_1h"] == 1     # one prior txn within the last hour
    assert out.loc[1, "amount_sum_24h"] == 100.0

def test_all_features_present():
    out = build_features(_df())
    assert set(FEATURES).issubset(out.columns)
