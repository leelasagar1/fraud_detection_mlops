# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train, Track, Register
# MAGIC Builds the training set from the feature table via a **point-in-time FeatureLookup**,
# MAGIC trains LightGBM with imbalance handling, and uses **`fe.log_model`** so the model
# MAGIC ships with its feature lookups baked in (auto feature retrieval at serving time in Phase 2).

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
TXN_TABLE     = f"{CATALOG}.{SCHEMA}.transactions"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.txn_features"
MODEL_NAME    = f"{CATALOG}.{SCHEMA}.fraud_clf"

# COMMAND ----------

# Labels carry only the keys + target; features are joined in by lookup (single source of truth).
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

labels = spark.table(TXN_TABLE).select("transaction_id", "event_ts", "is_fraud")

training_set = fe.create_training_set(
    df=labels,
    feature_lookups=[FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key="transaction_id",
        timestamp_lookup_key="event_ts",   # point-in-time correct join
    )],
    label="is_fraud",
    exclude_columns=["transaction_id", "customer_id", "event_ts"],
)
data = training_set.load_df().toPandas()

# COMMAND ----------

import mlflow, lightgbm as lgb, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# Time-based split — never random for temporal/fraud data.
data = data.sort_values("event_ts") if "event_ts" in data else data
for col in ["merchant_category", "channel"]:
    data[col] = data[col].astype("category")

FEATURES = [c for c in data.columns if c != "is_fraud"]
cut = int(len(data) * 0.8)
train, test = data.iloc[:cut], data.iloc[cut:]
Xtr, ytr = train[FEATURES], train["is_fraud"]
Xte, yte = test[FEATURES],  test["is_fraud"]
spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)   # imbalance correction

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run() as run:
    params = dict(objective="binary", n_estimators=300, learning_rate=0.05,
                  num_leaves=31, scale_pos_weight=spw)
    model = lgb.LGBMClassifier(**params).fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pr_auc, roc_auc = average_precision_score(yte, proba), roc_auc_score(yte, proba)

    # Threshold by business cost (beta>1 leans toward recall — catching fraud).
    prec, rec, thr = precision_recall_curve(yte, proba)
    beta = 2.0
    fbeta = (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-9)
    best_threshold = float(thr[np.nanargmax(fbeta[:-1])])

    mlflow.log_params(params)
    mlflow.log_metrics({"pr_auc": pr_auc, "roc_auc": roc_auc, "best_threshold": best_threshold})

    # fe.log_model packages the FeatureLookups => serving auto-fetches features in Phase 2.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.lightgbm,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
    )
    print(f"PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}  threshold={best_threshold:.3f}")

# COMMAND ----------

# Promote the new version with the modern alias pattern (replaces Staging/Production stages).
from mlflow.tracking import MlflowClient
c = MlflowClient()
latest = max(int(m.version) for m in c.search_model_versions(f"name='{MODEL_NAME}'"))
c.set_registered_model_alias(MODEL_NAME, "champion", latest)
print(f"{MODEL_NAME} v{latest} -> @champion")

# COMMAND ----------


