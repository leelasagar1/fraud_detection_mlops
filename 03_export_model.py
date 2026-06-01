# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Export the champion model for local serving
# MAGIC The champion was logged via `fe.log_model`, which wraps it in a feature-lookup layer
# MAGIC that needs the online store and won't run off-platform. We pull out the underlying
# MAGIC LightGBM model (same weights) and save a **portable** MLflow model to a UC Volume.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")   # harmless to be explicit; needed in some non-notebook contexts

# COMMAND ----------

CATALOG, SCHEMA = "main", "fraud"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_clf"
MODEL_URI  = f"models:/{MODEL_NAME}@champion"

import mlflow, os, shutil

# Try the native lightgbm flavor first; fall back to finding it nested in the FE wrapper.
model = None
try:
    model = mlflow.lightgbm.load_model(MODEL_URI)
    print("Loaded native lightgbm flavor at top level.")
except Exception as e:
    print("Top-level load failed, searching nested artifacts...\n", e)
    local = mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI)
    found = None
    for root, _, files in os.walk(local):
        if "MLmodel" in files:
            text = open(os.path.join(root, "MLmodel")).read()
            if "lightgbm" in text.lower():
                found = root
                print("Found nested lightgbm model at:", root)
    if not found:
        raise RuntimeError("No nested lightgbm flavor found — inspect printed paths.")
    model = mlflow.lightgbm.load_model(found)

# COMMAND ----------

import boto3
boto3.client("s3").get_object(
    Bucket="dbstorage-prod-o09ea",
    Key="uc/.../versions/e10d12db-.../MLmodel",
)

# COMMAND ----------

# Save a clean, portable MLflow model to a UC Volume, then zip it for browser download.
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.exports")
export_dir = f"/Volumes/{CATALOG}/{SCHEMA}/exports/fraud_model"
shutil.rmtree(export_dir, ignore_errors=True)
mlflow.lightgbm.save_model(model, export_dir)
shutil.make_archive(export_dir, "zip", root_dir=export_dir)
print("Saved + zipped -> download fraud_model.zip from Catalog > Volumes >",
      f"{CATALOG}.{SCHEMA}.exports")

# COMMAND ----------

# The two things your local service needs: the feature schema and the decision threshold.
c = mlflow.tracking.MlflowClient()
mv = c.get_model_version_by_alias(MODEL_NAME, "champion")
run = c.get_run(mv.run_id)
print("THRESHOLD (set this in the service):", run.data.metrics.get("best_threshold"))
try:
    print("FEATURE ORDER:", model.booster_.feature_name())
except Exception:
    print("FEATURE ORDER:", list(getattr(model, "feature_name_", [])))
