"""
RiseOS — Student Performance Prediction Model
Uses UCI Student Performance Dataset
Outputs: trained model + SHAP explainer
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib
import json

# ── Load UCI Student Performance Dataset ──────────────────────────────────────
# Download from: https://archive.ics.uci.edu/ml/datasets/student+performance
# Use student-mat.csv (Math) or student-por.csv (Portuguese)

def load_and_preprocess():
    df = pd.read_csv('student-mat.csv', sep=';')

    # Feature engineering
    # G3 = final grade (0-20), classify as Pass/Fail or Grade Band
    df['result'] = pd.cut(df['G3'],
        bins=[-1, 9, 13, 17, 20],
        labels=['Fail', 'Pass', 'Good', 'Excellent']
    )

    # Features we'll use (realistic for a student tracking app)
    features = [
        'studytime',    # weekly study time (1-4 scale)
        'failures',     # past class failures
        'absences',     # number of absences
        'G1',           # first period grade
        'G2',           # second period grade
        'health',       # health status 1-5
        'freetime',     # free time after school 1-5
        'goout',        # going out with friends 1-5
        'Dalc',         # workday alcohol consumption 1-5
        'Walc',         # weekend alcohol consumption 1-5
        'higher',       # wants higher education (yes/no)
        'internet',     # internet access (yes/no)
        'Medu',         # mother education 0-4
        'Fedu',         # father education 0-4
    ]

    # Encode binary categoricals
    df['higher']   = (df['higher']   == 'yes').astype(int)
    df['internet'] = (df['internet'] == 'yes').astype(int)

    X = df[features].copy()
    y = df['result']

    return X, y, features

def train():
    X, y, features = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   # Random Forest — robust, paper-worthy, and fully SHAP compatible
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # SHAP explainer for XAI
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature importance via SHAP
    # Feature importance via SHAP
    if isinstance(shap_values, list):
        mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
    elif shap_values.ndim == 3:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    importance = dict(zip(features, mean_shap.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\nTop Features (SHAP importance):")
    for feat, val in importance_sorted.items():
        print(f"  {feat:15s}: {val:.4f}")

    # Save artifacts
    joblib.dump(model, 'model.pkl')
    joblib.dump(explainer, 'explainer.pkl')
    with open('feature_importance.json', 'w') as f:
        json.dump(importance_sorted, f, indent=2)
    with open('features.json', 'w') as f:
        json.dump(features, f)
    with open('accuracy.json', 'w') as f:
        json.dump({'accuracy': round(acc, 4), 'n_train': len(X_train), 'n_test': len(X_test)}, f)

    print("\nModel saved: model.pkl")
    print("Explainer saved: explainer.pkl")
    print("Metrics saved: accuracy.json")

if __name__ == '__main__':
    train()
