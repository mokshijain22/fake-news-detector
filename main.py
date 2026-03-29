"""
Fake News Detection with SHAP Explainability  (v2.2 — WELFake)
Dataset : WELFake (72K articles, 4 combined sources)
Model   : XGBoost + TF-IDF + hand-crafted style features
Author  : Mokshi Jain

Fixes vs v2.1:
  • Dataset switched from ISOT to WELFake for better diversity
  • Model saved BEFORE SHAP so a SHAP crash never loses the model
  • SHAP wrapped in try/except — skipped gracefully if it fails
  • Metadata dataset label fixed to WELFake
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import scipy.sparse as sp

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
)
import xgboost as xgb
import shap
import joblib

from data_utils import load_welfake_dataset, preprocess_text, extract_extra_features
from visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap_summary,
    plot_shap_force,
    plot_top_features,
)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
MAX_FEATURES = 10_000
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
print("Loading dataset...")
df = load_welfake_dataset()
df["clean_text"] = df["text"].apply(preprocess_text)
print(f"  Total samples : {len(df):,}")
print(f"  Class balance :\n{df['label'].value_counts()}\n")

X_text = df["clean_text"]
y      = df["label"]

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ── 2. TF-IDF vectorisation ───────────────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
)
X_train_tfidf = tfidf.fit_transform(X_train_txt)
X_test_tfidf  = tfidf.transform(X_test_txt)

# ── 3. Extra style features ───────────────────────────────────────────────────
print("Extracting style features...")
X_train_extra = extract_extra_features(X_train_txt.reset_index(drop=True))
X_test_extra  = extract_extra_features(X_test_txt.reset_index(drop=True))

# Concatenate: sparse TF-IDF + dense style features → sparse matrix
X_train = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_extra)], format="csr")
X_test  = sp.hstack([X_test_tfidf,  sp.csr_matrix(X_test_extra)],  format="csr")

# ── 4. Train XGBoost with early stopping ─────────────────────────────────────
print("Training XGBoost classifier (with early stopping)...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print(f"  Best iteration : {model.best_iteration}")

# ── 5. Cross-validation ───────────────────────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation (TF-IDF only for speed)...")
cv_model = xgb.XGBClassifier(
    n_estimators=model.best_iteration or 100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(cv_model, X_train_tfidf, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
acc         = accuracy_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'='*50}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  CV Acc    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ── 7. Visualisations ─────────────────────────────────────────────────────────
print("\nGenerating visualizations...")
plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR)
plot_roc_curve(y_test, y_pred_prob, auc, OUTPUT_DIR)
plot_top_features(model, tfidf, OUTPUT_DIR)

# ── 8. Save model FIRST — before SHAP so a crash never loses the model ────────
print("\nSaving model...")
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))

metadata = {
    "accuracy":         round(acc * 100, 2),
    "roc_auc":          round(auc, 4),
    "cv_accuracy_mean": round(cv_scores.mean() * 100, 2),
    "cv_accuracy_std":  round(cv_scores.std() * 100, 2),
    "best_iteration":   int(model.best_iteration or 0),
    "n_features":       MAX_FEATURES,
    "extra_features":   5,
    "trained_on":       datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset":          "WELFake",   # ✅ fixed from ISOT
    "n_samples":        len(df),
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Model + metadata saved to {OUTPUT_DIR}/")

# ── 9. SHAP — optional, skipped gracefully if it fails ───────────────────────
print("\nComputing SHAP values (this may take ~1-2 min)...")
try:
    # ✅ Use full feature matrix (TF-IDF + style) to match training
    X_test_dense = sp.hstack([
        X_test_tfidf[:100],
        sp.csr_matrix(X_test_extra[:100])
    ]).toarray()
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_dense, check_additivity=False)
    plot_shap_summary(shap_values, X_test_dense, tfidf, OUTPUT_DIR)
    plot_shap_force(explainer, shap_values, X_test_dense, tfidf, idx=0, output_dir=OUTPUT_DIR)
    print("  SHAP plots saved.")
except Exception as e:
    print(f"  SHAP skipped: {e}")

print("\nDone! All plots saved to outputs/")