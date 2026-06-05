import os
import pandas as pd
import numpy as np
import dagshub
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. Inisialisasi DagsHub MLflow Remote Tracking
#dagshub.init(repo_owner='upan55678', repo_name='Breast_Cancer_MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/upan55678/Breast_Cancer_MLops.mlflow")
def train_baseline():
    # 2. Muat Dataset Siap Latih (Hasil Kriteria 1)
    dataset_path = "breast_cancer_clean.csv"
        
    df = pd.read_csv(dataset_path)
    
    # Pisahkan Fitur dan Target
    X = df.drop(columns=['Status'])
    y = df['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Setup Eksperimen MLflow
    mlflow.set_experiment("Breast_Cancer_Baseline")
    
    #with mlflow.start_run(run_name="RandomForest_Baseline"):
    with mlflow.start_run(run_name="RandomForest_Baseline", nested=True):
        # Parameter Model
        n_estimators = 50
        max_depth = 5
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Evaluasi Metrik
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # 4. MANUAL LOGGING 
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Simpan Model ke MLflow Registry
        mlflow.sklearn.log_model(model, "baseline_rf_model")
        
        print("Pelatihan Model Baseline Selesai!")
        print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_baseline()
