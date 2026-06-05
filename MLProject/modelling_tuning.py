import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Inisialisasi DagsHub
#dagshub.init(repo_owner='upan55678', repo_name='Breast_Cancer_MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/upan55678/Breast_Cancer_MLops.mlflow")
def train_with_tuning():
    dataset_path = "breast_cancer_clean.csv"
        
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['Status'])
    y = df['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.set_experiment("Breast_Cancer_Tuning")
    
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        # 1. Hyperparameter Tuning Setup
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [4, 8, 12],
            'criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # 2. Manual Logging Parameter Terbaik
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)
            
        # 3. Manual Logging Metrik Evaluasi
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # 4. GENERATE DAN LOG ARTEFAK TAMBAHAN (Syarat Advance: Minimal 2 Artefak)
        os.makedirs("./plots", exist_ok=True)
        
        # Artefak 1: Plot Confusion Matrix
        plt.figure(figsize=(5,4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Alive'], yticklabels=['Dead', 'Alive'])
        plt.title('Confusion Matrix - Tuned Model')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        cm_path = "./plots/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Artefak 2: Plot Feature Importance
        plt.figure(figsize=(6,4))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:] # Ambil 10 fitur teratas
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.title('Top 10 Feature Importances')
        feat_path = "./plots/feature_importance.png"
        plt.savefig(feat_path)
        plt.close()
        
        # Unggah artefak gambar ke server DagsHub MLflow
        mlflow.log_artifact(cm_path, "plots_evaluation")
        mlflow.log_artifact(feat_path, "plots_evaluation")
        
        # Simpan Model Akhir
        mlflow.sklearn.log_model(best_model, "tuned_rf_model")
        
        print("Tuning Model dan Pengiriman Artefak Selesai!")

if __name__ == "__main__":
    train_with_tuning()
