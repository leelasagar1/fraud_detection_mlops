# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup & Data
# MAGIC Creates the catalog/schema and writes a synthetic transaction stream to a Delta table.
# MAGIC Run on an **ML runtime** cluster.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config — change these to your workspace's catalog/schema
CATALOG = "main"
SCHEMA  = "fraud"
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
TXN_TABLE = f"{CATALOG}.{SCHEMA}.transactions"

# COMMAND ----------

# Synthetic generator (self-contained). `drift` lets you shift the distribution in
# Phase 3 to make monitoring/retraining demonstrate something real.
import numpy as np, pandas as pd

CATEGORIES = ["grocery","fuel","restaurant","retail","travel","electronics","online"]
CAT_RISK = {"grocery":.05,"fuel":.05,"restaurant":.07,"retail":.10,
            "travel":.20,"electronics":.25,"online":.30}

def generate(n_customers=2000, days=60, fraud_rate=0.002, drift=0.0, seed=42):
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "avg_amount": rng.lognormal(3.2, 0.6, n_customers),
        "active_hour": rng.integers(7, 22, n_customers),
    })
    w = np.array([CAT_RISK[c] for c in CATEGORIES])
    if drift: w = w * (1 + drift * rng.random(len(CATEGORIES)))
    w = w / w.sum()
    fraud_w = (w**2) / (w**2).sum()

    start = pd.Timestamp("2025-01-01")
    n = int(n_customers * days * 3)
    idx = rng.integers(0, n_customers, n)
    rows = []
    for i in range(n):
        c = cust.iloc[idx[i]]
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, days*86400)))
        fraud = rng.random() < fraud_rate
        if fraud:
            amt = c.avg_amount * rng.uniform(3, 12) * (1 + drift)
            hour = int(rng.choice([1,2,3,4])); cat = rng.choice(CATEGORIES, p=fraud_w)
            ch = "online"
        else:
            amt = max(1.0, rng.normal(c.avg_amount, c.avg_amount*0.4))
            hour = int(np.clip(rng.normal(c.active_hour, 2.5), 0, 23))
            cat = rng.choice(CATEGORIES, p=w); ch = rng.choice(["in_person","online"], p=[.7,.3])
        rows.append((i, ts.replace(hour=hour), int(c.customer_id),
                     round(float(amt),2), cat, ch, int(fraud)))
    return pd.DataFrame(rows, columns=["transaction_id","event_ts","customer_id",
                        "amount","merchant_category","channel","is_fraud"])

pdf = generate().sort_values("event_ts")
print(f"{len(pdf):,} txns, {pdf.is_fraud.mean():.3%} fraud")

# COMMAND ----------

(spark.createDataFrame(pdf)
      .write.mode("overwrite").saveAsTable(TXN_TABLE))
print(f"Wrote {TXN_TABLE}")
display(spark.table(TXN_TABLE).limit(5))

# COMMAND ----------


