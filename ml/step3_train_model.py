"""
NeuroRide – Step 3: Train Drowsiness Detection Model
=====================================================
Input  : neuroride_labeled.csv  (output of step2_label_data.py)
Output : neuroride_model.pkl    (trained Random Forest model)
         neuroride_scaler.pkl   (fitted scaler for preprocessing)
         neuroride_report.txt   (full evaluation report)

Run:
    python step3_train_model.py

Requirements:
    pip install scikit-learn
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.utils import resample

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "neuroride_labeled.csv"
MODEL_FILE   = "neuroride_model.pkl"
SCALER_FILE  = "neuroride_scaler.pkl"
REPORT_FILE  = "neuroride_report.txt"

FEATURE_COLS = [
    "Delta Waves", "Theta Waves",
    "Low Alpha Waves", "High Alpha Waves",
    "Low Beta Waves",  "High Beta Waves",
    "Low Gamma Waves", "High Gamma Waves",
    "Attention", "Meditation",
    "Theta_Beta_Ratio", "Alpha_Beta_Ratio",
    "Delta_AB_Ratio", "Total_Power",
]
TARGET_COL   = "Label"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  NeuroRide – Step 3: Model Training")
print("=" * 55)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"'{INPUT_FILE}' not found. Run step2_label_data.py first."
    )

df = pd.read_csv(INPUT_FILE)
print(f"\n[1] Loaded : {len(df):,} labeled rows")

alert_count  = (df[TARGET_COL] == 0).sum()
drowsy_count = (df[TARGET_COL] == 1).sum()
print(f"      Alert  (0): {alert_count:,}")
print(f"      Drowsy (1): {drowsy_count:,}")

# ── STEP A: Prepare features ──────────────────────────────────────────────────
X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# ── STEP B: Train / test split (stratified) ───────────────────────────────────
# Stratified keeps label ratio same in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\n[2] Split  : Train {len(X_train):,}  |  Test {len(X_test):,}  (80/20 stratified)")

# ── STEP C: Scale features ────────────────────────────────────────────────────
# StandardScaler: mean=0, std=1 for each feature
# Fit ONLY on train — never on test (prevents data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("[3] Features scaled (StandardScaler fit on train only)")

# ── STEP D: Balance training set (oversampling) ───────────────────────────────
# Since imbalanced-learn may not be installed, use sklearn's resample
# This duplicates minority (drowsy) samples to match majority (alert) count
train_df        = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
train_df["Label"] = y_train.values

alert_tr  = train_df[train_df["Label"] == 0]
drowsy_tr = train_df[train_df["Label"] == 1]

drowsy_upsampled = resample(
    drowsy_tr,
    replace=True,
    n_samples=len(alert_tr),
    random_state=RANDOM_STATE
)
balanced_train = pd.concat([alert_tr, drowsy_upsampled]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train_bal = balanced_train[FEATURE_COLS].values
y_train_bal = balanced_train["Label"].values

print(f"[4] Balanced train set: {len(X_train_bal):,} rows")
print(f"      Alert  : {(y_train_bal==0).sum():,}")
print(f"      Drowsy : {(y_train_bal==1).sum():,}")

# ── STEP E: Train Random Forest ───────────────────────────────────────────────
print("\n[5] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,       # number of trees — more = better, slower
    max_depth=None,         # trees grow fully unless limited
    min_samples_split=5,    # min samples to split a node
    min_samples_leaf=2,     # min samples at leaf — reduces overfitting
    class_weight="balanced",# extra safety against imbalance
    random_state=RANDOM_STATE,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train_bal, y_train_bal)
print("      Training complete!")

# ── STEP F: Evaluate on test set ──────────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=["Alert", "Drowsy"])

print(f"\n[6] Test Set Results:")
print(f"      Accuracy  : {accuracy*100:.2f}%")
print(f"      ROC-AUC   : {roc_auc:.4f}  (1.0 = perfect)")
print(f"\n      Confusion Matrix:")
print(f"                  Predicted")
print(f"                  Alert   Drowsy")
print(f"      Actual Alert  {cm[0][0]:5}   {cm[0][1]:5}")
print(f"      Actual Drowsy {cm[1][0]:5}   {cm[1][1]:5}")
print(f"\n      Classification Report:")
for line in report.split("\n"):
    print(f"      {line}")

# ── STEP G: Cross-validation ──────────────────────────────────────────────────
print("[7] Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Use unbalanced full dataset for honest CV
X_scaled_full = scaler.transform(X)
cv_scores = cross_val_score(model, X_scaled_full, y, cv=cv, scoring="f1", n_jobs=-1)
print(f"      F1 per fold : {[f'{s:.3f}' for s in cv_scores]}")
print(f"      Mean F1     : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── STEP H: Feature importance ────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

print(f"\n[8] Top feature importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 200)
    print(f"      {feat:<25}: {imp:.4f}  {bar}")

# ── STEP I: Save model & scaler ───────────────────────────────────────────────
with open(MODEL_FILE,  "wb") as f: pickle.dump(model,  f)
with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)
print(f"\n[9] Saved model  → {MODEL_FILE}")
print(f"    Saved scaler → {SCALER_FILE}")

# ── STEP J: Save full report ──────────────────────────────────────────────────
report_text = f"""
NeuroRide – Model Evaluation Report
=====================================
Dataset      : {len(df):,} rows  |  Alert: {alert_count:,}  |  Drowsy: {drowsy_count:,}
Model        : Random Forest (200 trees)
Test size    : {TEST_SIZE*100:.0f}%  ({len(X_test):,} rows)

Results
-------
Accuracy     : {accuracy*100:.2f}%
ROC-AUC      : {roc_auc:.4f}

Confusion Matrix:
             Predicted Alert   Predicted Drowsy
Actual Alert       {cm[0][0]:5}              {cm[0][1]:5}
Actual Drowsy      {cm[1][0]:5}              {cm[1][1]:5}

Classification Report:
{report}

Cross-Validation (5-fold F1):
{cv_scores}
Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}

Feature Importances (ranked):
{importances.to_string()}
"""

with open(REPORT_FILE, "w") as f:
    f.write(report_text)
print(f"    Saved report → {REPORT_FILE}")

# ── STEP K: Quick prediction demo ────────────────────────────────────────────
print(f"\n{'─'*55}")
print("  Quick prediction demo:")
sample = X_test_scaled[:3]
preds  = model.predict(sample)
probs  = model.predict_proba(sample)
label_map = {0: "Alert", 1: "Drowsy"}
for i, (pred, prob) in enumerate(zip(preds, probs)):
    print(f"  Sample {i+1}: {label_map[pred]}  "
          f"(Alert: {prob[0]*100:.1f}%  Drowsy: {prob[1]*100:.1f}%)")

print(f"\n{'─'*55}")
print(f"  Model training complete!")
print(f"  Next step: run step4_predict.py to test on new EEG data\n")
