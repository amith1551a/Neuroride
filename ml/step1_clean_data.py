"""
NeuroRide – Step 1: Data Cleaning & Preprocessing
===================================================
Input  : merged_output.csv  (raw EEG data from MindLink headband)
Output : neuroride_clean.csv (cleaned, ready for labeling)

Run:
    python step1_clean_data.py
"""

import pandas as pd
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "merged_output.csv"
OUTPUT_FILE = "neuroride_clean.csv"

WAVE_COLS = [
    "Delta Waves", "Theta Waves",
    "Low Alpha Waves", "High Alpha Waves",
    "Low Beta Waves",  "High Beta Waves",
    "Low Gamma Waves", "High Gamma Waves",
]
COGNITIVE_COLS = ["Attention", "Meditation"]
ALL_FEATURE_COLS = WAVE_COLS + COGNITIVE_COLS

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  NeuroRide – Step 1: Data Cleaning")
print("=" * 55)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"'{INPUT_FILE}' not found. "
        "Place it in the same folder as this script."
    )

df = pd.read_csv(INPUT_FILE)
print(f"\n[1] Loaded           : {len(df):,} rows")

# ── STEP A: Remove embedded summary rows ─────────────────────────────────────
# The MindLink software inserts "Average %" rows into the CSV
is_data_row = df["Date Time"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)
df = df[is_data_row].copy()
print(f"[2] After avg rows   : {len(df):,} rows  (removed summary rows)")

# ── STEP B: Parse timestamps ──────────────────────────────────────────────────
df["Date Time"] = pd.to_datetime(df["Date Time"], format="mixed")
df = df.sort_values("Date Time").reset_index(drop=True)

# ── STEP C: Remove bad signal rows ───────────────────────────────────────────
# Signal Strength > 0 means poor headband contact — unreliable data
df = df[df["Signal Strength"] == 0].copy()
print(f"[3] After bad signal : {len(df):,} rows  (Signal Strength = 0 only)")

# ── STEP D: Remove all-zero rows ─────────────────────────────────────────────
# Zeros = headset not yet initialized or completely lost contact
zero_mask = (df[ALL_FEATURE_COLS] == 0).all(axis=1)
df = df[~zero_mask].copy()
print(f"[4] After zero rows  : {len(df):,} rows  (removed all-zero readings)")

# ── STEP E: Handle missing values ────────────────────────────────────────────
missing_before = df[ALL_FEATURE_COLS].isnull().sum().sum()
if missing_before > 0:
    df[ALL_FEATURE_COLS] = df[ALL_FEATURE_COLS].fillna(
        df[ALL_FEATURE_COLS].median()
    )
    print(f"[5] Filled {missing_before} missing values with column medians")
else:
    print(f"[5] No missing values found")

# ── STEP F: Drop unused columns ───────────────────────────────────────────────
df = df.drop(columns=["Signal Strength"])

# ── STEP G: Compute derived features ─────────────────────────────────────────
# These ratios are well-known drowsiness indicators in EEG research

# Theta/Beta ratio – rises sharply during drowsiness
df["Theta_Beta_Ratio"] = df["Theta Waves"] / (
    df["Low Beta Waves"] + df["High Beta Waves"] + 1e-6
)

# Alpha/Beta ratio – increases as alertness drops
df["Alpha_Beta_Ratio"] = (
    df["Low Alpha Waves"] + df["High Alpha Waves"]
) / (df["Low Beta Waves"] + df["High Beta Waves"] + 1e-6)

# Delta/(Alpha+Beta) ratio – high in deep sleep
df["Delta_AB_Ratio"] = df["Delta Waves"] / (
    df["Low Alpha Waves"] + df["High Alpha Waves"] +
    df["Low Beta Waves"]  + df["High Beta Waves"]  + 1e-6
)

# Total power – sum of all bands (rough signal energy)
df["Total_Power"] = df[WAVE_COLS].sum(axis=1)

print(f"[6] Added 4 derived features (Theta/Beta, Alpha/Beta, Delta/AB ratios, Total Power)")

# ── STEP H: Add session date ──────────────────────────────────────────────────
df["Session_Date"] = df["Date Time"].dt.date
session_counts = df["Session_Date"].value_counts().sort_index()
print(f"\n[7] Sessions found:")
for date, count in session_counts.items():
    print(f"      {date}  →  {count:,} rows")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Final dataset      : {len(df):,} rows × {len(df.columns)} columns")
print(f"  Date range         : {df['Date Time'].min().date()} → {df['Date Time'].max().date()}")
print(f"  Features ready     : {ALL_FEATURE_COLS + ['Theta_Beta_Ratio','Alpha_Beta_Ratio','Delta_AB_Ratio','Total_Power']}")
print(f"{'─'*55}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved → {OUTPUT_FILE}")
print("  Next step: run step2_label_data.py\n")
