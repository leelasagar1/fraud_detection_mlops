# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Batch scoring (the thing we'll monitor)
# MAGIC No serving endpoint. We load the registered champion, score time-windowed batches,
# MAGIC and write predictions to Delta. In production this is a scheduled Workflow; here it's
# MAGIC a notebook. Labels (`is_fraud`) are written too, but treat them as ARRIVING LATER —
# MAGIC in real fraud you don't know the truth at scoring time.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm
# MAGIC dbutils.library.restartPython()
# MAGIC # then re-set registry URI and re-run

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
TXN_TABLE   = f"{CATALOG}.{SCHEMA}.transactions"
MODEL_NAME  = f"{CATALOG}.{SCHEMA}.fraud_clf"
PRED_TABLE  = f"{CATALOG}.{SCHEMA}.scored_predictions"
 
import mlflow, os, numpy as np, pandas as pd
 
# Load the underlying LightGBM from the champion (strip the FE wrapper, same as nb 03).
URI = f"models:/{MODEL_NAME}@champion"
try:
    clf = mlflow.lightgbm.load_model(URI)
except Exception:
    local = mlflow.artifacts.download_artifacts(artifact_uri=URI)
    path = next(r for r, _, f in os.walk(local)
                if "MLmodel" in f and "lightgbm" in open(os.path.join(r, "MLmodel")).read().lower())
    clf = mlflow.lightgbm.load_model(path)
 
FEATURES = ["amount", "amount_zscore", "txn_count_1h", "txn_count_24h",
            "amount_sum_24h", "merchant_category", "channel", "is_night"]

# COMMAND ----------

# Point-in-time-correct features in pandas (closed='left' excludes the current row).
def build_features(df: pd.DataFrame) -> pd.DataFrame:
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

def score(df: pd.DataFrame) -> pd.DataFrame:
    f = build_features(df)
    X = f[FEATURES].copy()
    for c in ["merchant_category", "channel"]:
        X[c] = X[c].astype("category")
    f["fraud_probability"] = clf.predict_proba(X)[:, 1]
    return f[["transaction_id", "customer_id", "event_ts",
              "fraud_probability", "is_fraud"] + FEATURES]

# COMMAND ----------

# Split history into weekly batches: earliest = monitoring REFERENCE, rest = "production".
pdf = spark.table(TXN_TABLE).toPandas()
pdf["event_ts"] = pd.to_datetime(pdf["event_ts"])
scored = score(pdf)
scored["batch"] = ("week_" +
    (scored["event_ts"].dt.isocalendar().week - scored["event_ts"].dt.isocalendar().week.min())
    .astype(int).astype(str).str.zfill(2))

(spark.createDataFrame(scored)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PRED_TABLE))
print(f"Wrote {PRED_TABLE}: {scored['batch'].nunique()} batches")
display(spark.sql(f"SELECT batch, count(*) n, avg(fraud_probability) avg_p FROM {PRED_TABLE} GROUP BY batch ORDER BY batch"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inject drift to demo monitoring (run after you've seen a clean baseline)
# MAGIC Re-import the generator from `data/generate_transactions.py`, make a DRIFTED batch,
# MAGIC score it, and append as batch `drifted`. Then re-run notebook 05 and watch PSI fire.

# COMMAND ----------

# MAGIC %pip install data

# COMMAND ----------

# from data.generate_transactions import generate          # or paste generate() inline
# drifted = generate(days=14, drift=0.6, seed=7)
# drifted["event_ts"] = pd.to_datetime(drifted["event_ts"])
# d = score(drifted); d["batch"] = "drifted"
# (spark.createDataFrame(d).write.mode("append").saveAsTable(PRED_TABLE))
# print("Appended drifted batch")

# COMMAND ----------


