# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Feature Engineering (point-in-time correct)
# MAGIC Builds velocity / amount-deviation / time-of-day features using ONLY past rows
# MAGIC per transaction, and writes them to a Unity Catalog feature table.
# MAGIC `rangeBetween(-seconds, -1)` excludes the current row → no label leakage.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
TXN_TABLE     = f"{CATALOG}.{SCHEMA}.transactions"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.txn_features"

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table(TXN_TABLE).withColumn("ts_unix", F.col("event_ts").cast("long"))

def past(seconds):
    return Window.partitionBy("customer_id").orderBy("ts_unix").rangeBetween(-seconds, -1)
w1h, w24h, w7d = past(3600), past(86400), past(604800)

feats = (df
    .withColumn("txn_count_1h",  F.count("*").over(w1h))
    .withColumn("txn_count_24h", F.count("*").over(w24h))
    .withColumn("amount_sum_24h", F.coalesce(F.sum("amount").over(w24h), F.lit(0.0)))
    .withColumn("amt_mean_7d", F.avg("amount").over(w7d))
    .withColumn("amt_std_7d",  F.coalesce(F.stddev("amount").over(w7d), F.lit(1.0)))
    .withColumn("amount_zscore",
        (F.col("amount") - F.col("amt_mean_7d")) / F.greatest(F.col("amt_std_7d"), F.lit(1.0)))
    .withColumn("is_night", (F.hour("event_ts") < 6).cast("int"))
    .na.fill({"amount_zscore": 0.0})
    .select("transaction_id", "customer_id", "event_ts", "amount", "amount_zscore",
            "txn_count_1h", "txn_count_24h", "amount_sum_24h",
            "merchant_category", "channel", "is_night")
)
display(feats.limit(5))

# COMMAND ----------

# Note: Feature Engineering APIs evolve — verify signatures against your runtime's
# `databricks-feature-engineering` version. timestamp_keys enables point-in-time joins.
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["transaction_id", "event_ts"],
    timestamp_keys=["event_ts"],
    df=feats,
    description="Point-in-time fraud features: velocity, amount deviation, time-of-day.",
)
print(f"Wrote {FEATURE_TABLE}")

# COMMAND ----------


