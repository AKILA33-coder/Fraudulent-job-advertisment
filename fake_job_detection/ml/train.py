"""
TrueHire AI - Model Training Script
Trains RandomForest + ExtraTrees + LogisticRegression ensemble
"""

import pandas as pd
import numpy as np
import pickle, os, json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(text):
    """Extract 7 fraud-detection features from raw job text."""
    import re
    t = str(text).lower()
    return {
        'has_fee':               int(bool(re.search(r'(registration|joining|training|application)\s*fee|pay.*fee|deposit', t))),
        'has_urgency':           int(bool(re.search(r'urgent|immediately|limited\s+slot|apply\s+now', t))),
        'has_whatsapp':          int('whatsapp' in t or 'telegram' in t),
        'no_email':              int(not bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))),
        'excessive_exclamation': int(text.count('!') > 5),
        'text_length':           len(text),
        'has_salary':            int(bool(re.search(r'salary|lpa|ctc|\$|inr|per\s*annum', t))),
    }

def build_feature_matrix(df):
    feats = df['description'].apply(extract_features)
    X = pd.DataFrame(list(feats))
    y = df['fraudulent'].astype(int)
    return X, y

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    csv_path = os.path.join(DATA_DIR, 'fake_jobs_raw.csv')
    if not os.path.exists(csv_path):
        print("Dataset not found. Running prepare_data.py first...")
        import subprocess, sys
        subprocess.run([sys.executable,
                        os.path.join(BASE, 'prepare_data.py')], check=True)
    df = pd.read_csv(csv_path)
    df['description'] = df['description'].fillna('')
    print("Loaded {} rows | Fake: {} | Legit: {}".format(
        len(df), df['fraudulent'].sum(), (df['fraudulent'] == 0).sum()))
    return df

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 60)
    print("  TrueHire AI - Model Training (RF + ET + LR Ensemble)")
    print("=" * 60)

    df = load_data()
    X, y = build_feature_matrix(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nSplit: {} train / {} test".format(len(X_train), len(X_test)))

    # Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # ── 1. Random Forest ──────────────────────────────────────────────────────
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_pred  = rf.predict(X_test_s)
    rf_proba = rf.predict_proba(X_test_s)[:, 1]

    results['random_forest'] = {
        'accuracy':  round(accuracy_score(y_test, rf_pred),  4),
        'precision': round(precision_score(y_test, rf_pred), 4),
        'recall':    round(recall_score(y_test, rf_pred),    4),
        'f1':        round(f1_score(y_test, rf_pred),        4),
        'roc_auc':   round(roc_auc_score(y_test, rf_proba),  4),
    }
    print("  Accuracy : {:.1f}%".format(results['random_forest']['accuracy']  * 100))
    print("  Precision: {:.1f}%".format(results['random_forest']['precision'] * 100))
    print("  Recall   : {:.1f}%".format(results['random_forest']['recall']    * 100))
    print("  F1-Score : {:.1f}%".format(results['random_forest']['f1']        * 100))
    print("  ROC-AUC  : {:.3f}".format(results['random_forest']['roc_auc']))

    cv_rf = cross_val_score(rf, X_train_s, y_train, cv=5, scoring='f1')
    print("  CV F1    : {:.1f}% +/- {:.1f}%".format(
        cv_rf.mean() * 100, cv_rf.std() * 100))

    # ── 2. Extra Trees Classifier ─────────────────────────────────────────────
    print("\n[2/3] Training Extra Trees Classifier...")
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    et.fit(X_train_s, y_train)
    et_pred  = et.predict(X_test_s)
    et_proba = et.predict_proba(X_test_s)[:, 1]

    results['extra_trees'] = {
        'accuracy':  round(accuracy_score(y_test, et_pred),  4),
        'precision': round(precision_score(y_test, et_pred), 4),
        'recall':    round(recall_score(y_test, et_pred),    4),
        'f1':        round(f1_score(y_test, et_pred),        4),
        'roc_auc':   round(roc_auc_score(y_test, et_proba),  4),
    }
    print("  Accuracy : {:.1f}%".format(results['extra_trees']['accuracy']  * 100))
    print("  Precision: {:.1f}%".format(results['extra_trees']['precision'] * 100))
    print("  Recall   : {:.1f}%".format(results['extra_trees']['recall']    * 100))
    print("  F1-Score : {:.1f}%".format(results['extra_trees']['f1']        * 100))
    print("  ROC-AUC  : {:.3f}".format(results['extra_trees']['roc_auc']))

    cv_et = cross_val_score(et, X_train_s, y_train, cv=5, scoring='f1')
    print("  CV F1    : {:.1f}% +/- {:.1f}%".format(
        cv_et.mean() * 100, cv_et.std() * 100))

    # ── 3. Logistic Regression ────────────────────────────────────────────────
    print("\n[3/3] Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred  = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]

    results['logistic_regression'] = {
        'accuracy':  round(accuracy_score(y_test, lr_pred),  4),
        'precision': round(precision_score(y_test, lr_pred), 4),
        'recall':    round(recall_score(y_test, lr_pred),    4),
        'f1':        round(f1_score(y_test, lr_pred),        4),
        'roc_auc':   round(roc_auc_score(y_test, lr_proba),  4),
    }
    print("  Accuracy : {:.1f}%".format(results['logistic_regression']['accuracy']  * 100))
    print("  F1-Score : {:.1f}%".format(results['logistic_regression']['f1']        * 100))

    # ── Ensemble (ET 40% + RF 35% + LR 25%) ──────────────────────────────────
    ensemble_proba = (et_proba * 0.40) + (rf_proba * 0.35) + (lr_proba * 0.25)
    ensemble_pred  = (ensemble_proba >= 0.5).astype(int)

    results['ensemble'] = {
        'accuracy':  round(accuracy_score(y_test,  ensemble_pred),  4),
        'precision': round(precision_score(y_test, ensemble_pred),  4),
        'recall':    round(recall_score(y_test,    ensemble_pred),  4),
        'f1':        round(f1_score(y_test,        ensemble_pred),  4),
        'roc_auc':   round(roc_auc_score(y_test,   ensemble_proba), 4),
    }
    print("\nEnsemble (ET 40% + RF 35% + LR 25%):")
    print("  Accuracy : {:.1f}%".format(results['ensemble']['accuracy']  * 100))
    print("  Precision: {:.1f}%".format(results['ensemble']['precision'] * 100))
    print("  Recall   : {:.1f}%".format(results['ensemble']['recall']    * 100))
    print("  F1-Score : {:.1f}%".format(results['ensemble']['f1']        * 100))
    print("  ROC-AUC  : {:.3f}".format(results['ensemble']['roc_auc']))

    # ── Feature importance (from ET) ──────────────────────────────────────────
    feat_names  = list(X.columns)
    fi_df = pd.DataFrame({
        'feature':    feat_names,
        'importance': et.feature_importances_
    }).sort_values('importance', ascending=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)

    print("\nFeature Importance (Extra Trees):")
    for _, row in fi_df.iterrows():
        bar = '#' * int(row['importance'] * 40)
        print("  {:<28s} {} {:.3f}".format(row['feature'], bar, row['importance']))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, ensemble_pred)
    print("\nConfusion Matrix (Ensemble):")
    print("              Pred:Legit  Pred:Fake")
    print("  Act:Legit   {:>9}  {:>9}".format(cm[0][0], cm[0][1]))
    print("  Act:Fake    {:>9}  {:>9}".format(cm[1][0], cm[1][1]))

    # ── Save all models ───────────────────────────────────────────────────────
    print("\nSaving models...")
    with open(os.path.join(MODEL_DIR, 'random_forest.pkl'),       'wb') as f: pickle.dump(rf,     f)
    with open(os.path.join(MODEL_DIR, 'extra_trees.pkl'),         'wb') as f: pickle.dump(et,     f)
    with open(os.path.join(MODEL_DIR, 'logistic_regression.pkl'), 'wb') as f: pickle.dump(lr,     f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'),              'wb') as f: pickle.dump(scaler, f)

    metrics = {
        'trained_at':    datetime.now().isoformat(),
        'model_version': '2.0',
        'models':        ['RandomForest', 'ExtraTrees', 'LogisticRegression'],
        'ensemble_weights': {'extra_trees': 0.40, 'random_forest': 0.35, 'logistic_regression': 0.25},
        'dataset_size':  len(df),
        'results':       results,
        'features':      feat_names,
    }
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("  random_forest.pkl        saved")
    print("  extra_trees.pkl          saved")
    print("  logistic_regression.pkl  saved")
    print("  scaler.pkl               saved")
    print("  metrics.json             saved")
    print("  feature_importance.csv   saved")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Ensemble Accuracy : {:.1f}%".format(results['ensemble']['accuracy'] * 100))
    print("  Run: python app_ml.py")
    print("=" * 60 + "\n")
    return results

if __name__ == '__main__':
    train()
