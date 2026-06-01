# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Monitoring: drift + prediction drift + quality decay
# MAGIC Implemented from first principles (PSI + KS) so you know what the libraries do.
# MAGIC Three questions every batch monitor must answer:
# MAGIC  1. **Feature drift** — did the inputs shift? (PSI per feature, KS for numeric)
# MAGIC  2. **Prediction drift** — did the model's output distribution shift?
# MAGIC  3. **Quality decay** — once labels arrive, is PR-AUC dropping?

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
PRED_TABLE  = f"{CATALOG}.{SCHEMA}.scored_predictions"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.monitoring_metrics"

import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

NUMERIC = ["amount", "amount_zscore", "txn_count_1h", "txn_count_24h", "amount_sum_24h"]
CATEG   = ["merchant_category", "channel"]

# PSI: <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant drift (the alert line).
def psi_numeric(ref, cur, bins=10):
    q = np.quantile(ref, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, q)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, q)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))

def psi_categorical(ref, cur):
    cats = set(ref) | set(cur)
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(1e-6)
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).clip(1e-6)
    return float(np.sum((c - r) * np.log(c / r)))

# COMMAND ----------

df = spark.table(PRED_TABLE).toPandas()
batches = sorted(df["batch"].unique())
ref_name, ref = batches[0], df[df.batch == batches[0]]
print(f"Reference batch: {ref_name} ({len(ref)} rows)")

rows = []
for b in batches[1:]:
    cur = df[df.batch == b]
    # 1. feature drift
    for f in NUMERIC:
        rows.append((b, "feature_psi", f, psi_numeric(ref[f], cur[f])))
        rows.append((b, "feature_ks_pvalue", f, float(ks_2samp(ref[f], cur[f]).pvalue)))
    for f in CATEG:
        rows.append((b, "feature_psi", f, psi_categorical(ref[f], cur[f])))
    # 2. prediction drift
    rows.append((b, "prediction_psi", "fraud_probability",
                 psi_numeric(ref["fraud_probability"], cur["fraud_probability"])))
    # 3. quality (labels treated as delayed-but-now-available)
    if cur["is_fraud"].nunique() > 1:
        rows.append((b, "pr_auc", "model",
                     float(average_precision_score(cur["is_fraud"], cur["fraud_probability"]))))

m = pd.DataFrame(rows, columns=["batch", "metric", "feature", "value"])

# COMMAND ----------

# Alerting rules
PSI_ALERT = 0.2
ref_pr_auc = average_precision_score(ref["is_fraud"], ref["fraud_probability"])

def alert(r):
    if r.metric in ("feature_psi", "prediction_psi"):
        return "DRIFT" if r.value > PSI_ALERT else "ok"
    if r.metric == "pr_auc":
        return "QUALITY_DROP" if r.value < 0.8 * ref_pr_auc else "ok"
    return "ok"

m["alert"] = m.apply(alert, axis=1)
m["checked_at"] = pd.Timestamp.utcnow()

(spark.createDataFrame(m)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(METRICS_TABLE))

# COMMAND ----------

print(f"Reference PR-AUC: {ref_pr_auc:.3f}\n")
fired = m[m.alert != "ok"]
if len(fired):
    print("ALERTS:")
    print(fired[["batch", "metric", "feature", "value", "alert"]].to_string(index=False))
else:
    print("No alerts — all batches within thresholds.")

# Trend the headline metrics in a Databricks SQL dashboard on monitoring_metrics, e.g.:
#   SELECT batch, value FROM main.fraud.monitoring_metrics
#   WHERE metric='pr_auc' ORDER BY batch
display(spark.table(METRICS_TABLE).orderBy("metric", "feature", "batch"))

# COMMAND ----------

try: dbutils.jobs.taskValues.set("retrain_needed", bool(len(fired)))
except Exception: pass

# COMMAND ----------


