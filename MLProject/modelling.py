import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import numpy as npAdd commentMore actions
import time
import os
import sys
import json

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
@@ -68,6 +70,13 @@
    max_err = max_error(y_test, y_pred)
    training_time = end - start

    # Set tags (agar muncul folder 'tags/')
    mlflow.set_tags({
        "project": "Titanic Survival",
        "author": "Fakhrizal",
        "type": "regression"
    })

    # Log params & metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("degree_poly", 2)
@@ -79,19 +88,45 @@
    mlflow.log_metric("Max_Error", max_err)
    mlflow.log_metric("Training_Time", training_time)

    # Log model sebagai artifact biasa (model.pkl)
    joblib.dump(pipeline, "model.pkl")
    mlflow.log_artifact("model.pkl")

    # Log model dengan mlflow.sklearn agar muncul folder model/
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    # Simpan hasil prediksi ke CSV
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    pred_df.to_csv("cleaned_data.csv", index=False)
    mlflow.log_artifact("cleaned_data.csv")

    # Simpan metrik dalam format JSON
    metrics_dict = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Explained_Variance": explained_var,
        "Max_Error": max_err,
        "Training_Time": training_time
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    mlflow.log_artifact("metrics.json")

    # Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")

    # Print metrics
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}, Max Error: {max_err:.4f}, Training Time: {training_time:.2f}s")

# Save local model
output_path = "./Models/Forest_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Model saved to: {output_path}")
