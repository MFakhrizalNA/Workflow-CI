import sys, os, time, argparse
import mlflow
import joblib
import pandas as pd
import numpy as np
import dagshub
import dagshub.auth
import requests
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to cleaned dataset CSV file')
args = parser.parse_args()
csv_path = os.path.abspath(args.data_path)

# Path ke preprocessing class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

# == SETUP TRACKING ==========================================================
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def setup_tracking():
    try:
        # Uji koneksi ke DagsHub
        test_url = "https://dagshub.com/MFakhrizalNA/MSML_Fakhrizal.mlflow/api/2.0/mlflow/experiments/list"
        resp = requests.get(test_url, auth=(username, password), timeout=10)
        if resp.status_code >= 500:
            raise ConnectionError("💥 Server DagsHub error!")

        # Setup MLflow ke DagsHub
        dagshub.auth.add_app_token(password)
        dagshub.init(repo_owner='MFakhrizalNA', repo_name='MSML_Fakhrizal', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/MFakhrizalNA/MSML_Fakhrizal.mlflow")
        mlflow.set_experiment("Titanic Survival Prediction 1")
        print("✅ Tracking MLflow ke DagsHub")
    except Exception as e:
        print(f"⚠️ Gagal koneksi ke DagsHub: {e}")
        print("🔁 Tracking fallback ke lokal (file:///tmp/mlruns)")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("Titanic Survival Prediction (Local)")

setup_tracking()

# Aktifkan autologging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# Load dataset
data = pd.read_csv(csv_path)
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing dan pipeline
preprocessor = SklearnPreprocessor(
    num_columns=['Age', 'Fare'],
    ordinal_columns=['Pclass'],
    nominal_columns=['Sex', 'Embarked'],
    degree=2
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Mulai autologging run
with mlflow.start_run(run_name="Autolog - RandomForestClassifier"):
    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()

    # Prediksi dan evaluasi
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    training_time = end - start
    support = y_test.value_counts().to_dict()

    # Logging dataset sebagai artefak
    mlflow.log_artifact(csv_path, artifact_path="dataset")

    # Tampilkan hasil
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Training Time: {training_time:.2f} detik")
    print(f"Support (1): {support.get(1, 0)}, Support (0): {support.get(0, 0)}")

# Simpan model ke lokal
output_path = "models/forest_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Model disimpan di: {output_path}")
