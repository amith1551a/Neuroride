# NeuroRide — Full Project Design

**Date:** 2026-05-30  
**Status:** Approved  
**Scope:** Python ML pipeline (Steps 1–5) + Flutter Android app (Step 5A–5E)

---

## Overview

NeuroRide detects rider drowsiness in real time by analyzing EEG brainwave signals from a FT&S MindLink headband (NeuroSky TGAM1 chipset, Bluetooth Classic SPP). A trained Random Forest model runs on-device via TFLite and triggers audio/vibration alerts when drowsiness is detected.

---

## Repository Structure

```
Neuroride/
  data/
    merged_output.csv          ← raw EEG data (provided)
  ml/
    step1_clean_data.py
    step2_label_data.py
    step3_train_model.py
    step4_predict.py
    step5_convert_tflite.py
    neuroride_clean.csv        ← generated
    neuroride_labeled.csv      ← generated
    neuroride_model.pkl        ← generated
    neuroride_scaler.pkl       ← generated
    neuroride_report.txt       ← generated
    neuroride_model.tflite     ← generated
  app/                         ← Flutter project root
    lib/
      models/eeg_data.dart
      services/bluetooth_service.dart
      services/thinkgear_parser.dart
      services/alert_service.dart
      services/tflite_service.dart
      providers/eeg_provider.dart
      screens/splash_screen.dart
      screens/device_scan_screen.dart
      screens/dashboard_screen.dart
      main.dart
    assets/
      neuroride_model.tflite
      alert_sound.mp3
    pubspec.yaml
    android/AndroidManifest.xml
  docs/
    superpowers/specs/
      2026-05-30-neuroride-design.md
```

---

## Part 1 — Python ML Pipeline

### Step 1 — Data Cleaning (`ml/step1_clean_data.py`)

**Input:** `data/merged_output.csv`  
**Output:** `ml/neuroride_clean.csv`

Operations:

- Remove rows where `Signal Strength > 0` (bad contact)
- Remove rows where `Date Time` contains "Average %" (summary rows embedded by the FT&S software)
- Remove rows where all wave values are 0 (initialization rows)
- Add 4 derived features:
  - `Theta_Beta_Ratio = Theta / (Low_Beta + High_Beta + 0.001)`
  - `Alpha_Beta_Ratio = (Low_Alpha + High_Alpha) / (Low_Beta + High_Beta + 0.001)`
  - `Delta_AB_Ratio = Delta / (Low_Alpha + High_Alpha + 0.001)`
  - `Total_Power = sum of all 8 wave bands`

Expected output: ~7,173 clean rows.

### Step 2 — Labeling (`ml/step2_label_data.py`)

**Input:** `ml/neuroride_clean.csv`  
**Output:** `ml/neuroride_labeled.csv`

Rule-based multi-score labeling. Each row scores points for drowsiness indicators:

- Theta > 4 → +1
- Delta > 5 → +1
- Attention < 40 → +1
- Low Beta < 2 → +1
- Theta_Beta_Ratio > 2 → +1
- Meditation > 70 → +1

Label: `Drowsy (1)` if score ≥ 3, else `Alert (0)`.

Expected: ~22–25% drowsy, ~75–78% alert.

### Step 3 — Model Training (`ml/step3_train_model.py`)

**Input:** `ml/neuroride_labeled.csv`  
**Output:** `ml/neuroride_model.pkl`, `ml/neuroride_scaler.pkl`, `ml/neuroride_report.txt`

- Features: all 8 wave bands + Attention + Meditation + 4 derived = 14 features
- StandardScaler normalization
- SMOTE oversampling to balance classes
- Random Forest (100 trees, random_state=42)
- Evaluation: accuracy, F1, ROC-AUC, confusion matrix, feature importance
- 5-fold cross-validation

Expected: ~99% accuracy, ~0.998 ROC-AUC.

### Step 4 — Real-time Prediction (`ml/step4_predict.py`)

**Input:** `ml/neuroride_model.pkl`, `ml/neuroride_scaler.pkl`

Two modes:

- **Mode 1 — Stream simulation:** replay `neuroride_labeled.csv` row by row, raise alarm after 3 consecutive drowsy readings
- **Mode 2 — Single reading:** manual EEG value input → instant prediction

Alert thresholds:

- ✅ SAFE: drowsy probability < 35%
- ⚠️ WARNING: 35–50%
- 🚨 DANGER: > 50%

### Step 5 — TFLite Conversion (`ml/step5_convert_tflite.py`)

**Input:** `ml/neuroride_model.pkl`, `ml/neuroride_scaler.pkl`  
**Output:** `ml/neuroride_model.tflite`

Conversion path: sklearn RandomForest → ONNX (via `sklearn-onnx`) → TFLite (via `onnx-tf` + TF Lite Converter). The scaler mean/scale arrays are embedded as a JSON metadata file alongside the TFLite model so Flutter can normalize inputs identically.

---

## Part 2 — Flutter Mobile App

### Hardware

- **Device:** FT&S MindLink EEG Headband
- **Chipset:** NeuroSky TGAM1
- **Protocol:** Bluetooth Classic (SPP), NOT BLE
- **Packet format:** ThinkGear binary protocol
  - Sync bytes: `0xAA 0xAA`
  - Length byte
  - Payload (8 wave bands as 3-byte big-endian uint32, eSense Attention/Meditation as 1-byte)
  - Checksum byte

### Flutter Package Selection

| Package                    | Version | Purpose                                    |
| -------------------------- | ------- | ------------------------------------------ |
| `flutter_bluetooth_serial` | ^0.4.0  | Bluetooth Classic SPP (required for TGAM1) |
| `tflite_flutter`           | ^0.10.4 | On-device TFLite inference                 |
| `audioplayers`             | ^5.2.1  | Alert sound playback                       |
| `vibration`                | ^1.8.4  | Haptic alert                               |
| `provider`                 | ^6.1.2  | State management                           |
| `fl_chart`                 | ^0.66.2 | Live EEG waveform chart                    |
| `permission_handler`       | ^11.3.1 | Bluetooth + location permissions           |
| `hive_flutter`             | ^1.1.0  | Local session history storage              |

### App Architecture

```
main.dart
  └── MultiProvider
        ├── EegProvider (ChangeNotifier)
        │     ├── BluetoothService (connects, streams bytes)
        │     ├── ThinkGearParser (bytes → EegData)
        │     └── TfliteService (EegData → drowsy probability)
        └── AlertService (probability → sound + vibration)

Screens:
  SplashScreen → requests permissions, checks BT state
  DeviceScanScreen → lists paired devices, tap to connect
  DashboardScreen → live chart, drowsiness meter, alert banner
```

### Data Flow

```
MindLink headband
  → Bluetooth Classic SPP
  → BluetoothService (raw bytes stream)
  → ThinkGearParser (packet validation + decode)
  → EegData model (8 bands + attention + meditation)
  → TfliteService (normalize → infer → probability)
  → EegProvider (notifies UI + feeds AlertService)
  → DashboardScreen (chart + meter)
  → AlertService (if 3 consecutive DANGER readings → alarm)
```

### Alert Logic

- Maintain rolling buffer of last 3 predictions
- If all 3 are DANGER (probability > 0.50): trigger alarm
- Alarm: plays `alert_sound.mp3` + continuous vibration
- Auto-dismiss after 10 seconds or user tap
- Alert levels displayed in UI:
  - ✅ Green — SAFE (< 35%)
  - ⚠️ Yellow — WARNING (35–50%)
  - 🚨 Red — DANGER (> 50%)

### Android Permissions (`AndroidManifest.xml`)

```xml
<uses-permission android:name="android.permission.BLUETOOTH"/>
<uses-permission android:name="android.permission.BLUETOOTH_ADMIN"/>
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT"/>
<uses-permission android:name="android.permission.BLUETOOTH_SCAN"/>
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
<uses-permission android:name="android.permission.VIBRATE"/>
```

---

## Error Handling

- Bad signal (Signal Strength > 0): parser skips packet, UI shows "Poor Signal" banner
- BT disconnection: auto-reconnect attempt ×3, then show reconnect dialog
- Model inference failure: fall back to SAFE state, log error
- SMOTE failure (too few minority samples): fall back to class_weight='balanced' in RandomForest

---

## Out of Scope

- iOS support (Bluetooth Classic is Android-only in Flutter)
- Cloud sync / Firebase
- Multi-rider personalization
- GPS / ride analytics
