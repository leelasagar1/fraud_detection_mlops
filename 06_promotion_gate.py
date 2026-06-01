# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Promotion gate (champion vs challenger)
# MAGIC Trains a challenger, evaluates it AND the incumbent champion on the SAME recent
# MAGIC holdout, and promotes the challenger to `@champion` ONLY if it wins by a margin.
# MAGIC This is model CD: promotion is earned by metrics, not granted by running.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.txn_features"
TXN_TABLE     = f"{CATALOG}.{SCHEMA}.transactions"
MODEL_NAME    = f"{CATALOG}.{SCHEMA}.fraud_clf"
MARGIN = 0.005     # challenger must beat champion PR-AUC by at least this much

import mlflow, os, lightgbm as lgb
from sklearn.metrics import average_precision_score
from mlflow.tracking import MlflowClient

FEATURES = ["amount", "amount_zscore", "txn_count_1h", "txn_count_24h",
            "amount_sum_24h", "merchant_category", "channel", "is_night"]

def load_lgb(uri):
    try:
        return mlflow.lightgbm.load_model(uri)
    except Exception:
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        p = next(r for r, _, f in os.walk(local)
                 if "MLmodel" in f and "lightgbm" in open(os.path.join(r, "MLmodel")).read().lower())
        return mlflow.lightgbm.load_model(p)

# COMMAND ----------

# Read engineered features + labels straight from the feature table (single source of truth).
data = spark.sql(f"""
    SELECT f.event_ts, f.amount, f.amount_zscore, f.txn_count_1h, f.txn_count_24h,
           f.amount_sum_24h, f.merchant_category, f.channel, f.is_night, t.is_fraud
    FROM {FEATURE_TABLE} f JOIN {TXN_TABLE} t USING (transaction_id)
""").toPandas().sort_values("event_ts")
for c in ["merchant_category", "channel"]:
    data[c] = data[c].astype("category")

cut = int(len(data) * 0.8)
train, hold = data.iloc[:cut], data.iloc[cut:]
yh = hold["is_fraud"]

# COMMAND ----------

# Train challenger on the older slice.
spw = (train.is_fraud == 0).sum() / max((train.is_fraud == 1).sum(), 1)
challenger = lgb.LGBMClassifier(objective="binary", n_estimators=300,
                                learning_rate=0.05, num_leaves=31, scale_pos_weight=spw)
challenger.fit(train[FEATURES], train["is_fraud"])

# Evaluate incumbent + challenger on the SAME holdout.
champion = load_lgb(f"models:/{MODEL_NAME}@champion")
champ_auc = average_precision_score(yh, champion.predict_proba(hold[FEATURES])[:, 1])
chal_auc  = average_precision_score(yh, challenger.predict_proba(hold[FEATURES])[:, 1])
print(f"champion PR-AUC={champ_auc:.4f}   challenger PR-AUC={chal_auc:.4f}   margin={MARGIN}")

# COMMAND ----------

promoted = chal_auc > champ_auc + MARGIN
if promoted:
    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run():
        mlflow.log_metrics({"holdout_pr_auc": chal_auc, "incumbent_pr_auc": champ_auc})
        mlflow.lightgbm.log_model(challenger, name="model",
                                  input_example=hold[FEATURES].head(2),
                                  registered_model_name=MODEL_NAME)
    c = MlflowClient()
    v = max(int(m.version) for m in c.search_model_versions(f"name='{MODEL_NAME}'"))
    c.set_registered_model_alias(MODEL_NAME, "champion", v)
    print(f"PROMOTED: challenger -> v{v} @champion  (+{chal_auc - champ_auc:.4f})")
else:
    print("REJECTED: challenger did not clear the margin. Champion unchanged.")

# Expose the decision to the orchestrating job (ignored on interactive runs).
try:
    dbutils.jobs.taskValues.set("promoted", bool(promoted))
except Exception:
    pass

# COMMAND ----------


