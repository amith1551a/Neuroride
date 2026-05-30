"""
NeuroRide – Step 2: Label Data (Drowsy vs Alert)
=================================================
Input  : neuroride_clean.csv   (output of step1_clean_data.py)
Output : neuroride_labeled.csv (with Label column: 0=Alert, 1=Drowsy)

Labeling strategy: Multi-rule scoring based on EEG research
Each condition that indicates drowsiness adds 1 point (drowsy score).
Final label: score >= 3 → Drowsy (1), else → Alert (0)

Run:
    python step2_label_data.py
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "neuroride_clean.csv"
OUTPUT_FILE = "neuroride_labeled.csv"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  NeuroRide – Step 2: Labeling Data")
print("=" * 55)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"'{INPUT_FILE}' not found. Run step1_clean_data.py first."
    )

df = pd.read_csv(INPUT_FILE)
df["Date Time"] = pd.to_datetime(df["Date Time"], format="mixed")
print(f"\n[1] Loaded : {len(df):,} clean rows")

# ── STEP A: Clip extreme ratio outliers ───────────────────────────────────────
# Ratios can spike to millions when Beta waves are near zero
# Cap at 99th percentile to prevent outliers from dominating
for col in ["Theta_Beta_Ratio", "Alpha_Beta_Ratio", "Delta_AB_Ratio"]:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

print("[2] Clipped ratio outliers at 99th percentile")

# ── STEP B: Rule-based drowsiness scoring ────────────────────────────────────
#
# Each rule is based on established EEG drowsiness research:
#
# Rule 1 – High Delta Waves (>= p75 = 4.0)
#           Delta dominates during deep sleep / low consciousness
#
# Rule 2 – High Theta Waves (>= p75 = 3.0)
#           Theta rises during transition from alert to drowsy
#
# Rule 3 – Low Attention (<= 35)
#           MindLink proprietary metric — drops significantly when drowsy
#
# Rule 4 – High Meditation (>= 70)
#           Paradoxically high during drowsiness (deep relaxation state)
#
# Rule 5 – Theta/Beta Ratio >= p90 (2.5)
#           Classic EEG drowsiness marker — most cited in literature
#
# Rule 6 – Low Beta activity (Low + High Beta <= 4.0 combined)
#           Beta suppression = reduced active thinking = drowsiness

df["rule_delta"]     = (df["Delta Waves"] >= 4.0).astype(int)
df["rule_theta"]     = (df["Theta Waves"] >= 3.0).astype(int)
df["rule_attention"] = (df["Attention"] <= 35).astype(int)
df["rule_meditation"]= (df["Meditation"] >= 70).astype(int)
df["rule_tb_ratio"]  = (df["Theta_Beta_Ratio"] >= 2.5).astype(int)
df["rule_low_beta"]  = ((df["Low Beta Waves"] + df["High Beta Waves"]) <= 4.0).astype(int)

rule_cols = [
    "rule_delta", "rule_theta", "rule_attention",
    "rule_meditation", "rule_tb_ratio", "rule_low_beta"
]

df["Drowsy_Score"] = df[rule_cols].sum(axis=1)

print("\n[3] Drowsy score distribution (0=fully alert → 6=fully drowsy):")
score_counts = df["Drowsy_Score"].value_counts().sort_index()
for score, count in score_counts.items():
    bar = "█" * int(count / 50)
    print(f"      Score {score}: {count:5,}  {bar}")

# ── STEP C: Assign final labels ───────────────────────────────────────────────
# Threshold: score >= 3 means at least 3 drowsiness indicators are active
DROWSY_THRESHOLD = 3
df["Label"] = (df["Drowsy_Score"] >= DROWSY_THRESHOLD).astype(int)
# 0 = Alert, 1 = Drowsy

alert_count  = (df["Label"] == 0).sum()
drowsy_count = (df["Label"] == 1).sum()
total        = len(df)

print(f"\n[4] Labels assigned (threshold = {DROWSY_THRESHOLD}/6 rules):")
print(f"      Alert  (0): {alert_count:,}  ({100*alert_count/total:.1f}%)")
print(f"      Drowsy (1): {drowsy_count:,}  ({100*drowsy_count/total:.1f}%)")

# ── STEP D: Per-session label breakdown ───────────────────────────────────────
print("\n[5] Label breakdown per session:")
for date, group in df.groupby("Session_Date"):
    a = (group["Label"] == 0).sum()
    d = (group["Label"] == 1).sum()
    print(f"      {date}  →  Alert: {a:,}  |  Drowsy: {d:,}")

# ── STEP E: Rule contribution analysis ───────────────────────────────────────
print("\n[6] How often each rule fired (% of all rows):")
rule_labels = {
    "rule_delta"     : "High Delta (>= 4.0)",
    "rule_theta"     : "High Theta (>= 3.0)",
    "rule_attention" : "Low Attention (<= 35)",
    "rule_meditation": "High Meditation (>= 70)",
    "rule_tb_ratio"  : "Theta/Beta Ratio >= 2.5",
    "rule_low_beta"  : "Low Beta activity (<= 4.0)",
}
for col, label in rule_labels.items():
    pct = 100 * df[col].sum() / len(df)
    print(f"      {label:<35}: {pct:.1f}%")

# ── STEP F: Drop intermediate rule columns ───────────────────────────────────
df = df.drop(columns=rule_cols + ["Drowsy_Score"])

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'─'*55}")
print(f"  Saved → {OUTPUT_FILE}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  ⚠  Note: Drowsy samples ({drowsy_count:,}) are fewer than Alert")
print(f"     ({alert_count:,}). Step 3 will handle this imbalance with SMOTE.")
print(f"\n  Next step: run step3_train_model.py\n")
