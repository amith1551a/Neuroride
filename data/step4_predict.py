"""
NeuroRide – Step 4: Real-time Drowsiness Prediction
=====================================================
This script simulates real-time EEG prediction the way your
mobile app will work — reading one EEG sample at a time and
instantly predicting Alert or Drowsy.

Two modes:
  1. LIVE mode   — reads row-by-row from a CSV (simulates Bluetooth stream)
  2. SINGLE mode — predict on one manually entered EEG reading

Input  : neuroride_model.pkl   (trained model from step3)
         neuroride_scaler.pkl  (fitted scaler from step3)
         neuroride_labeled.csv (used in LIVE mode to simulate stream)

Run:
    python step4_predict.py
"""

import pickle
import time
import os
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_FILE       = "neuroride_model.pkl"
SCALER_FILE      = "neuroride_scaler.pkl"
DATA_FILE        = "neuroride_labeled.csv"

FEATURE_COLS = [
    "Delta Waves", "Theta Waves",
    "Low Alpha Waves", "High Alpha Waves",
    "Low Beta Waves",  "High Beta Waves",
    "Low Gamma Waves", "High Gamma Waves",
    "Attention", "Meditation",
    "Theta_Beta_Ratio", "Alpha_Beta_Ratio",
    "Delta_AB_Ratio", "Total_Power",
]

# Alert threshold — if drowsy probability >= this, trigger alert
# Lower = more sensitive (more alerts), Higher = stricter
ALERT_THRESHOLD = 0.50

# In LIVE mode: seconds between each reading (set to 0 for instant)
STREAM_DELAY = 0.3

# Number of consecutive drowsy readings before raising alarm
# Prevents false alerts from single noisy readings
CONSECUTIVE_DROWSY_LIMIT = 3

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    for f in [MODEL_FILE, SCALER_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"'{f}' not found. Run step3_train_model.py first."
            )
    with open(MODEL_FILE,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
    return model, scaler

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def compute_features(row: dict) -> pd.DataFrame:
    """
    Takes a raw EEG reading dict and returns a feature-ready DataFrame.
    This mirrors exactly what step1 did — must be identical for
    the model to work correctly on new data.
    """
    delta = row.get("Delta Waves", 0)
    theta = row.get("Theta Waves", 0)
    low_a = row.get("Low Alpha Waves", 0)
    hi_a  = row.get("High Alpha Waves", 0)
    low_b = row.get("Low Beta Waves", 0)
    hi_b  = row.get("High Beta Waves", 0)
    low_g = row.get("Low Gamma Waves", 0)
    hi_g  = row.get("High Gamma Waves", 0)
    attn  = row.get("Attention", 0)
    medit = row.get("Meditation", 0)

    beta_total  = low_b + hi_b + 1e-6
    alpha_total = low_a + hi_a
    ab_total    = alpha_total + beta_total

    features = {
        "Delta Waves"      : delta,
        "Theta Waves"      : theta,
        "Low Alpha Waves"  : low_a,
        "High Alpha Waves" : hi_a,
        "Low Beta Waves"   : low_b,
        "High Beta Waves"  : hi_b,
        "Low Gamma Waves"  : low_g,
        "High Gamma Waves" : hi_g,
        "Attention"        : attn,
        "Meditation"       : medit,
        "Theta_Beta_Ratio" : theta / beta_total,
        "Alpha_Beta_Ratio" : alpha_total / beta_total,
        "Delta_AB_Ratio"   : delta / (ab_total + 1e-6),
        "Total_Power"      : delta + theta + low_a + hi_a + low_b + hi_b + low_g + hi_g,
    }
    return pd.DataFrame([features])[FEATURE_COLS]

# ── PREDICT ONE SAMPLE ────────────────────────────────────────────────────────
def predict(model, scaler, eeg_reading: dict) -> dict:
    """
    Takes a single EEG reading (dict) and returns prediction result.
    Returns:
        label       : "Alert" or "Drowsy"
        confidence  : drowsy probability (0.0 – 1.0)
        alert_level : "SAFE" / "WARNING" / "DANGER"
    """
    X = compute_features(eeg_reading)
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    drowsy_prob = probs[1]

    if drowsy_prob < 0.35:
        alert_level = "SAFE"
    elif drowsy_prob < ALERT_THRESHOLD:
        alert_level = "WARNING"
    else:
        alert_level = "DANGER"

    return {
        "label"       : "Drowsy" if pred == 1 else "Alert",
        "confidence"  : round(float(drowsy_prob), 4),
        "alert_level" : alert_level,
        "alert_pct"   : f"{drowsy_prob*100:.1f}%",
        "alert_pct_"  : f"{probs[0]*100:.1f}%",
    }

# ── DISPLAY RESULT ────────────────────────────────────────────────────────────
def display_result(result: dict, sample_num: int = None, timestamp: str = None):
    icons = {"SAFE": "✅", "WARNING": "⚠️ ", "DANGER": "🚨"}
    icon  = icons[result["alert_level"]]

    prefix = f"[{sample_num:04d}]" if sample_num else "[----]"
    ts     = f"  {timestamp}" if timestamp else ""

    print(
        f"{prefix}{ts}  "
        f"{icon} {result['alert_level']:<8}  "
        f"State: {result['label']:<7}  "
        f"Drowsy: {result['alert_pct']:<6}  "
        f"Alert: {result['alert_pct_']}"
    )

# ── MODE 1: LIVE STREAM SIMULATION ────────────────────────────────────────────
def run_live_mode(model, scaler):
    if not os.path.exists(DATA_FILE):
        print(f"  '{DATA_FILE}' not found — switching to single mode.")
        run_single_mode(model, scaler)
        return

    df = pd.read_csv(DATA_FILE)
    total = len(df)
    print(f"\n  Streaming {total:,} EEG samples from {DATA_FILE}...")
    print(f"  Delay per sample : {STREAM_DELAY}s  |  Alert threshold: {ALERT_THRESHOLD}")
    print(f"  Consecutive drowsy alarm after: {CONSECUTIVE_DROWSY_LIMIT} readings")
    print(f"\n  {'Sample':<8} {'Timestamp':<26} {'Status':<10} {'State':<9} {'Drowsy':>8} {'Alert':>8}")
    print(f"  {'─'*75}")

    consecutive_drowsy = 0
    drowsy_count = 0
    alert_count  = 0
    alarm_count  = 0

    for i, (_, row) in enumerate(df.iterrows()):
        eeg = {col: row[col] for col in FEATURE_COLS
               if col in ["Delta Waves","Theta Waves","Low Alpha Waves",
                          "High Alpha Waves","Low Beta Waves","High Beta Waves",
                          "Low Gamma Waves","High Gamma Waves","Attention","Meditation"]}

        result = predict(model, scaler, eeg)
        ts = str(row.get("Date Time", ""))[:19]
        display_result(result, sample_num=i+1, timestamp=ts)

        if result["label"] == "Drowsy":
            drowsy_count += 1
            consecutive_drowsy += 1
            if consecutive_drowsy >= CONSECUTIVE_DROWSY_LIMIT:
                alarm_count += 1
                print(f"\n  {'!'*55}")
                print(f"  🚨  ALARM: {CONSECUTIVE_DROWSY_LIMIT} consecutive drowsy readings!")
                print(f"  🔔  BUZZER / VIBRATION TRIGGERED — Wake up!")
                print(f"  {'!'*55}\n")
                consecutive_drowsy = 0  # reset after alarm
        else:
            alert_count += 1
            consecutive_drowsy = 0

        time.sleep(STREAM_DELAY)

    # Summary
    print(f"\n  {'─'*55}")
    print(f"  Session Summary")
    print(f"  {'─'*55}")
    print(f"  Total samples  : {total:,}")
    print(f"  Alert readings : {alert_count:,}  ({100*alert_count/total:.1f}%)")
    print(f"  Drowsy readings: {drowsy_count:,}  ({100*drowsy_count/total:.1f}%)")
    print(f"  Alarms raised  : {alarm_count}")

# ── MODE 2: SINGLE PREDICTION ─────────────────────────────────────────────────
def run_single_mode(model, scaler):
    print("\n  Enter EEG values (press Enter to use default/typical values):")
    print("  Values are on the MindLink's 0–100 scale.\n")

    defaults = {
        "Delta Waves"     : 2.0,
        "Theta Waves"     : 2.0,
        "Low Alpha Waves" : 4.0,
        "High Alpha Waves": 3.0,
        "Low Beta Waves"  : 5.0,
        "High Beta Waves" : 6.0,
        "Low Gamma Waves" : 2.0,
        "High Gamma Waves": 1.0,
        "Attention"       : 55.0,
        "Meditation"      : 50.0,
    }

    eeg = {}
    for key, default in defaults.items():
        try:
            val = input(f"  {key:<22} (default {default}): ").strip()
            eeg[key] = float(val) if val else default
        except ValueError:
            eeg[key] = default

    result = predict(model, scaler, eeg)

    print(f"\n  {'─'*40}")
    print(f"  Prediction Result")
    print(f"  {'─'*40}")
    print(f"  State       : {result['label']}")
    print(f"  Alert level : {result['alert_level']}")
    print(f"  Drowsy prob : {result['alert_pct']}")
    print(f"  Alert prob  : {result['alert_pct_']}")

    if result["alert_level"] == "DANGER":
        print(f"\n  🚨 ALERT: Rider is drowsy! Trigger buzzer now!")
    elif result["alert_level"] == "WARNING":
        print(f"\n  ⚠️  WARNING: Drowsiness developing. Monitor closely.")
    else:
        print(f"\n  ✅ Rider is alert and safe.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NeuroRide – Step 4: Real-time Prediction")
    print("=" * 55)

    print("\n  Loading model...")
    model, scaler = load_model()
    print("  Model ready!\n")

    print("  Select mode:")
    print("  1. Live stream simulation (replay your collected data)")
    print("  2. Single reading prediction (enter values manually)")
    choice = input("\n  Enter 1 or 2: ").strip()

    if choice == "1":
        run_live_mode(model, scaler)
    else:
        run_single_mode(model, scaler)

if __name__ == "__main__":
    main()
