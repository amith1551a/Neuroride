# NeuroRide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild NeuroRide from scratch — Python ML pipeline that trains a drowsiness detection model from EEG data, exports it to TFLite, and a Flutter Android app that connects to the FT&S MindLink headband via Bluetooth Classic, runs on-device inference, and alerts the rider in real time.

**Architecture:** Python pipeline processes `data/merged_output.csv` → cleans → labels → trains RandomForest → exports TFLite + scaler JSON. Flutter app reads ThinkGear binary packets from MindLink over Bluetooth Classic SPP, normalizes EEG values, runs TFLite inference, and triggers audio+vibration alarm after 3 consecutive danger readings.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, imbalanced-learn, TensorFlow 2.15, Flutter 3.x, flutter_bluetooth_serial, tflite_flutter, provider, fl_chart, audioplayers, vibration, permission_handler

---

## File Map

### ML Pipeline (run on PC once to produce model files)
| File | Responsibility |
|---|---|
| `ml/requirements.txt` | Python dependencies |
| `ml/step1_clean_data.py` | Remove bad rows, add 4 derived features → `neuroride_clean.csv` |
| `ml/step2_label_data.py` | Multi-rule drowsiness labeling → `neuroride_labeled.csv` |
| `ml/step3_train_model.py` | Train Random Forest + SMOTE, save pkl files + report |
| `ml/step4_predict.py` | Stream simulation + manual single-reading prediction |
| `ml/step5_convert_tflite.py` | Train TF Keras model, export TFLite + scaler_params.json |

### Flutter App (`app/`)
| File | Responsibility |
|---|---|
| `app/pubspec.yaml` | Package dependencies + asset declarations |
| `app/android/app/src/main/AndroidManifest.xml` | Bluetooth + location + vibration permissions |
| `app/lib/models/eeg_data.dart` | EEG reading data class + 14-feature vector computation |
| `app/lib/services/thinkgear_parser.dart` | TGAM1 binary packet decoder |
| `app/lib/services/bluetooth_service.dart` | Bluetooth Classic SPP connection + byte stream |
| `app/lib/services/tflite_service.dart` | TFLite model loader + inference → PredictionResult |
| `app/lib/services/alert_service.dart` | Audio + vibration alarm |
| `app/lib/providers/eeg_provider.dart` | State management: connects all services, alarm logic |
| `app/lib/screens/splash_screen.dart` | Permission requests + BT state check |
| `app/lib/screens/device_scan_screen.dart` | List paired devices, connect |
| `app/lib/screens/dashboard_screen.dart` | Live waveform chart + drowsiness gauge + alarm banner |
| `app/lib/main.dart` | App entry point, provider wiring |

---

## Part 1 — Python ML Pipeline

---

### Task 1: ML folder setup + dependencies

**Files:**
- Create: `ml/requirements.txt`
- Create: `data/` (copy CSV here)

- [ ] **Step 1: Create folders**

```bash
mkdir ml
mkdir data
```

Copy `merged_output.csv` into `data/` folder.

- [ ] **Step 2: Create `ml/requirements.txt`**

```
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
imbalanced-learn==0.12.0
joblib==1.3.2
tensorflow==2.15.0
matplotlib==3.8.2
```

- [ ] **Step 3: Install**

```bash
pip install -r ml/requirements.txt
```

- [ ] **Step 4: Verify**

```bash
python -c "import pandas, sklearn, imblearn, tensorflow; print('All packages OK')"
```

Expected: `All packages OK`

- [ ] **Step 5: Commit**

```bash
git add ml/requirements.txt
git commit -m "chore: add Python ML requirements"
```

---

### Task 2: Step 1 — Data Cleaning

**Files:**
- Create: `ml/step1_clean_data.py`

- [ ] **Step 1: Create `ml/step1_clean_data.py`**

```python
import pandas as pd
import numpy as np


def load_and_clean(input_path='data/merged_output.csv',
                   output_path='ml/neuroride_clean.csv'):
    print("Loading data...")
    df = pd.read_csv(input_path, header=0)
    df.columns = [
        'Signal Strength', 'Date Time', 'Delta Waves', 'Theta Waves',
        'Low Alpha Waves', 'High Alpha Waves', 'Low Beta Waves',
        'High Beta Waves', 'Low Gamma Waves', 'High Gamma Waves',
        'Attention', 'Meditation'
    ]
    print(f"Original rows: {len(df)}")

    # Remove embedded "Average %" summary rows (Signal Strength is NaN/non-numeric)
    df = df[pd.to_numeric(df['Signal Strength'], errors='coerce').notna()].copy()
    df['Signal Strength'] = df['Signal Strength'].astype(float)
    print(f"After removing summary rows: {len(df)}")

    # Remove bad signal rows (Signal Strength > 0 means poor electrode contact)
    df = df[df['Signal Strength'] == 0.0].copy()
    print(f"After removing bad signal rows: {len(df)}")

    # Convert all numeric columns
    wave_cols = [
        'Delta Waves', 'Theta Waves', 'Low Alpha Waves', 'High Alpha Waves',
        'Low Beta Waves', 'High Beta Waves', 'Low Gamma Waves', 'High Gamma Waves'
    ]
    for col in wave_cols + ['Attention', 'Meditation']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=wave_cols + ['Attention', 'Meditation'])

    # Remove all-zero initialization rows
    zero_mask = (df[wave_cols].sum(axis=1) == 0) & (df['Attention'] == 0)
    df = df[~zero_mask].copy()
    print(f"After removing zero rows: {len(df)}")

    # Add 4 derived features (proven drowsiness indicators)
    df['Theta_Beta_Ratio'] = df['Theta Waves'] / (
        df['Low Beta Waves'] + df['High Beta Waves'] + 0.001)
    df['Alpha_Beta_Ratio'] = (df['Low Alpha Waves'] + df['High Alpha Waves']) / (
        df['Low Beta Waves'] + df['High Beta Waves'] + 0.001)
    df['Delta_AB_Ratio'] = df['Delta Waves'] / (
        df['Low Alpha Waves'] + df['High Alpha Waves'] + 0.001)
    df['Total_Power'] = df[wave_cols].sum(axis=1)

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} with {len(df)} rows and {len(df.columns)} columns")
    return df


if __name__ == '__main__':
    load_and_clean()
```

- [ ] **Step 2: Run**

```bash
python ml/step1_clean_data.py
```

Expected output:
```
Original rows: 7383
After removing summary rows: 7376
After removing bad signal rows: 7184
After removing zero rows: 7173
Saved ml/neuroride_clean.csv with 7173 rows and 16 columns
```

- [ ] **Step 3: Verify**

```bash
python -c "
import pandas as pd
df = pd.read_csv('ml/neuroride_clean.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Nulls:', df.isnull().sum().sum())
"
```

Expected: shape ~(7173, 16), 0 nulls.

- [ ] **Step 4: Commit**

```bash
git add ml/step1_clean_data.py
git commit -m "feat: add data cleaning script Step 1"
```

---

### Task 3: Step 2 — Labeling

**Files:**
- Create: `ml/step2_label_data.py`

- [ ] **Step 1: Create `ml/step2_label_data.py`**

```python
import pandas as pd


def label_data(input_path='ml/neuroride_clean.csv',
               output_path='ml/neuroride_labeled.csv'):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Multi-rule scoring: each rule adds 1 drowsiness point
    score = pd.Series(0, index=df.index)
    score += (df['Theta Waves'] > 4).astype(int)        # high theta = light sleep/drowsy
    score += (df['Delta Waves'] > 5).astype(int)         # high delta = deep sleep
    score += (df['Attention'] < 40).astype(int)          # low attention = drowsy
    score += (df['Low Beta Waves'] < 2).astype(int)      # low beta = low alertness
    score += (df['Theta_Beta_Ratio'] > 2).astype(int)    # classic EEG drowsiness marker
    score += (df['Meditation'] > 70).astype(int)         # high relaxation = drowsy

    # Label drowsy if 3 or more rules fire
    df['Label'] = (score >= 3).astype(int)

    alert_n  = (df['Label'] == 0).sum()
    drowsy_n = (df['Label'] == 1).sum()
    print(f"Alert:  {alert_n:5d} ({alert_n / len(df) * 100:.1f}%)")
    print(f"Drowsy: {drowsy_n:5d} ({drowsy_n / len(df) * 100:.1f}%)")

    print("\nRule activation rates:")
    rules = [
        ('Theta > 4',       df['Theta Waves'] > 4),
        ('Delta > 5',       df['Delta Waves'] > 5),
        ('Attention < 40',  df['Attention'] < 40),
        ('Low Beta < 2',    df['Low Beta Waves'] < 2),
        ('Theta/Beta > 2',  df['Theta_Beta_Ratio'] > 2),
        ('Meditation > 70', df['Meditation'] > 70),
    ]
    for name, mask in rules:
        print(f"  {name:20s}: {mask.mean()*100:.1f}%")

    print("\nSession breakdown:")
    df['_date'] = pd.to_datetime(df['Date Time']).dt.date
    for date, grp in df.groupby('_date'):
        a = (grp['Label'] == 0).sum()
        d = (grp['Label'] == 1).sum()
        print(f"  {date}: Alert={a}, Drowsy={d}")
    df.drop(columns=['_date'], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")
    return df


if __name__ == '__main__':
    label_data()
```

- [ ] **Step 2: Run**

```bash
python ml/step2_label_data.py
```

Expected: Alert ~77%, Drowsy ~23%.

- [ ] **Step 3: Verify**

```bash
python -c "
import pandas as pd
df = pd.read_csv('ml/neuroride_labeled.csv')
print(df['Label'].value_counts())
"
```

Expected: Label 0 → ~5500+, Label 1 → ~1600+

- [ ] **Step 4: Commit**

```bash
git add ml/step2_label_data.py
git commit -m "feat: add rule-based labeling script Step 2"
```

---

### Task 4: Step 3 — Model Training

**Files:**
- Create: `ml/step3_train_model.py`

- [ ] **Step 1: Create `ml/step3_train_model.py`**

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

FEATURES = [
    'Delta Waves', 'Theta Waves', 'Low Alpha Waves', 'High Alpha Waves',
    'Low Beta Waves', 'High Beta Waves', 'Low Gamma Waves', 'High Gamma Waves',
    'Attention', 'Meditation',
    'Theta_Beta_Ratio', 'Alpha_Beta_Ratio', 'Delta_AB_Ratio', 'Total_Power'
]


def train(input_path='ml/neuroride_labeled.csv'):
    df = pd.read_csv(input_path)
    X = df[FEATURES].values
    y = df['Label'].values
    print(f"Loaded {len(df)} rows | Drowsy: {y.sum()} | Alert: {(y==0).sum()}")

    # Scale first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE on training set only to avoid data leakage
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_res)} training samples")

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = (y_pred == y_test).mean()
    roc_auc  = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Alert', 'Drowsy']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5-fold cross-validation
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    print(f"\nCross-val F1: {cv.mean():.3f} ± {cv.std():.3f}")

    # Feature importance
    importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 5 features:")
    for feat, imp in importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    # Save model and scaler
    joblib.dump(model,  'ml/neuroride_model.pkl')
    joblib.dump(scaler, 'ml/neuroride_scaler.pkl')

    # Save text report
    report = (
        f"NeuroRide Model Report\n"
        f"======================\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"ROC-AUC  : {roc_auc:.4f}\n"
        f"CV F1    : {cv.mean():.3f} ± {cv.std():.3f}\n\n"
        f"{classification_report(y_test, y_pred, target_names=['Alert','Drowsy'])}\n"
        f"Feature Importance:\n"
        + "\n".join(f"  {f}: {i:.4f}" for f, i in importance)
    )
    with open('ml/neuroride_report.txt', 'w') as fh:
        fh.write(report)

    print("\nSaved: neuroride_model.pkl, neuroride_scaler.pkl, neuroride_report.txt")
    return model, scaler


if __name__ == '__main__':
    train()
```

- [ ] **Step 2: Run**

```bash
python ml/step3_train_model.py
```

Expected: Accuracy ~99%, ROC-AUC ~0.999

- [ ] **Step 3: Verify model loads**

```bash
python -c "
import joblib
m = joblib.load('ml/neuroride_model.pkl')
s = joblib.load('ml/neuroride_scaler.pkl')
print('Model estimators:', len(m.estimators_))
print('Scaler mean shape:', s.mean_.shape)
"
```

Expected: `Model estimators: 100`, `Scaler mean shape: (14,)`

- [ ] **Step 4: Commit**

```bash
git add ml/step3_train_model.py
git commit -m "feat: add Random Forest model training script Step 3"
```

---

### Task 5: Step 4 — Prediction Script

**Files:**
- Create: `ml/step4_predict.py`

- [ ] **Step 1: Create `ml/step4_predict.py`**

```python
import pandas as pd
import numpy as np
import joblib

FEATURES = [
    'Delta Waves', 'Theta Waves', 'Low Alpha Waves', 'High Alpha Waves',
    'Low Beta Waves', 'High Beta Waves', 'Low Gamma Waves', 'High Gamma Waves',
    'Attention', 'Meditation',
    'Theta_Beta_Ratio', 'Alpha_Beta_Ratio', 'Delta_AB_Ratio', 'Total_Power'
]

SAFE_THRESHOLD    = 0.35
DANGER_THRESHOLD  = 0.50
ALARM_CONSECUTIVE = 3


def _add_derived(row: dict) -> dict:
    d = dict(row)
    d['Theta_Beta_Ratio'] = d['Theta Waves'] / (d['Low Beta Waves'] + d['High Beta Waves'] + 0.001)
    d['Alpha_Beta_Ratio'] = (d['Low Alpha Waves'] + d['High Alpha Waves']) / (
        d['Low Beta Waves'] + d['High Beta Waves'] + 0.001)
    d['Delta_AB_Ratio']   = d['Delta Waves'] / (d['Low Alpha Waves'] + d['High Alpha Waves'] + 0.001)
    d['Total_Power']      = sum(d[c] for c in [
        'Delta Waves','Theta Waves','Low Alpha Waves','High Alpha Waves',
        'Low Beta Waves','High Beta Waves','Low Gamma Waves','High Gamma Waves'])
    return d


def predict_single(eeg: dict, model, scaler) -> dict:
    X = scaler.transform([[eeg[f] for f in FEATURES]])
    prob = model.predict_proba(X)[0][1]
    if prob < SAFE_THRESHOLD:
        level = '✅ SAFE'
    elif prob < DANGER_THRESHOLD:
        level = '⚠️ WARNING'
    else:
        level = '🚨 DANGER'
    return {'probability': prob, 'level': level, 'is_drowsy': prob >= DANGER_THRESHOLD}


def mode1_stream(model, scaler, csv_path='ml/neuroride_labeled.csv'):
    print("\n=== MODE 1: Stream Simulation ===")
    df = pd.read_csv(csv_path)
    consecutive = 0
    for i, row in df.iterrows():
        eeg = _add_derived(row.to_dict())
        result = predict_single(eeg, model, scaler)
        if result['is_drowsy']:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= ALARM_CONSECUTIVE:
            print(f"Row {i:5d}: {result['level']} ({result['probability']:.1%})  ← 🔔 ALARM!")
            consecutive = 0
        elif i % 100 == 0:
            print(f"Row {i:5d}: {result['level']} ({result['probability']:.1%})")
    print("Simulation complete.")


def mode2_single(model, scaler):
    print("\n=== MODE 2: Single Reading ===")
    def g(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return float(val) if val else float(default)

    eeg = {
        'Delta Waves':       g("Delta",       5.0),
        'Theta Waves':       g("Theta",       8.0),
        'Low Alpha Waves':   g("Low Alpha",   3.0),
        'High Alpha Waves':  g("High Alpha",  2.0),
        'Low Beta Waves':    g("Low Beta",    3.0),
        'High Beta Waves':   g("High Beta",   4.0),
        'Low Gamma Waves':   g("Low Gamma",   2.0),
        'High Gamma Waves':  g("High Gamma",  1.0),
        'Attention':         g("Attention",  48.0),
        'Meditation':        g("Meditation", 57.0),
    }
    eeg = _add_derived(eeg)
    result = predict_single(eeg, model, scaler)
    print(f"\nResult: {result['level']}  ({result['probability']:.1%} drowsy probability)")


if __name__ == '__main__':
    print("Loading model...")
    model  = joblib.load('ml/neuroride_model.pkl')
    scaler = joblib.load('ml/neuroride_scaler.pkl')
    print("1 - Stream simulation\n2 - Single reading")
    choice = input("Choice [1/2]: ").strip()
    if choice == '2':
        mode2_single(model, scaler)
    else:
        mode1_stream(model, scaler)
```

- [ ] **Step 2: Run (stream mode)**

```bash
echo "1" | python ml/step4_predict.py
```

Expected: Rows print with SAFE/WARNING/DANGER labels; occasional ALARM! lines.

- [ ] **Step 3: Commit**

```bash
git add ml/step4_predict.py
git commit -m "feat: add real-time prediction script Step 4"
```

---

### Task 6: Step 5 — TFLite Conversion

**Files:**
- Create: `ml/step5_convert_tflite.py`

Note: We train a small TF/Keras model on the same labeled data and export it directly to TFLite. This is more reliable than converting sklearn → ONNX → TFLite.

- [ ] **Step 1: Create `ml/step5_convert_tflite.py`**

```python
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'Delta Waves', 'Theta Waves', 'Low Alpha Waves', 'High Alpha Waves',
    'Low Beta Waves', 'High Beta Waves', 'Low Gamma Waves', 'High Gamma Waves',
    'Attention', 'Meditation',
    'Theta_Beta_Ratio', 'Alpha_Beta_Ratio', 'Delta_AB_Ratio', 'Total_Power'
]


def convert(input_path='ml/neuroride_labeled.csv'):
    df = pd.read_csv(input_path)
    X = df[FEATURES].values.astype(np.float32)
    y = df['Label'].values.astype(np.float32)
    print(f"Loaded {len(df)} rows | Drowsy: {int(y.sum())} | Alert: {int((y==0).sum())}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Save scaler params as JSON so Flutter app can normalize inputs
    scaler_params = {
        'mean':          scaler.mean_.tolist(),
        'scale':         scaler.scale_.tolist(),
        'feature_names': FEATURES,
    }
    with open('ml/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("Saved ml/scaler_params.json")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights to handle imbalance
    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    weight = {0: 1.0, 1: float(n_neg / n_pos)}
    print(f"Class weight for drowsy: {weight[1]:.2f}")

    # Small 3-layer Keras model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor='val_auc', mode='max'
    )
    model.fit(
        X_train, y_train,
        epochs=60, batch_size=32,
        class_weight=weight,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  AUC: {auc:.4f}")

    # Convert to TFLite with default optimisation (quantisation-aware)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open('ml/neuroride_model.tflite', 'wb') as f:
        f.write(tflite_bytes)
    print(f"Saved ml/neuroride_model.tflite ({len(tflite_bytes):,} bytes)")

    # Quick sanity-check: run one inference through the TFLite interpreter
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    sample = X_test[:1]
    interp.set_tensor(inp[0]['index'], sample)
    interp.invoke()
    prob = float(interp.get_tensor(out[0]['index'])[0][0])
    print(f"Sanity check prediction: {prob:.3f} (actual label: {int(y_test[0])})")
    print("\nInput shape :", inp[0]['shape'])
    print("Output shape:", out[0]['shape'])


if __name__ == '__main__':
    convert()
```

- [ ] **Step 2: Run**

```bash
python ml/step5_convert_tflite.py
```

Expected: Training progress printed, then:
```
Saved ml/scaler_params.json
Saved ml/neuroride_model.tflite (XXXX bytes)
Sanity check prediction: 0.XXX (actual label: 0 or 1)
Input shape : [1 14]
Output shape: [1  1]
```

- [ ] **Step 3: Verify both files exist**

```bash
python -c "
import os, json
print('tflite size:', os.path.getsize('ml/neuroride_model.tflite'), 'bytes')
d = json.load(open('ml/scaler_params.json'))
print('features:', len(d['feature_names']))
print('first 3 means:', d['mean'][:3])
"
```

Expected: size > 0, features: 14

- [ ] **Step 4: Copy model files to Flutter assets**

```bash
cp ml/neuroride_model.tflite app/assets/neuroride_model.tflite
cp ml/scaler_params.json     app/assets/scaler_params.json
```

- [ ] **Step 5: Commit**

```bash
git add ml/step5_convert_tflite.py app/assets/
git commit -m "feat: add TFLite conversion script Step 5 + copy assets"
```

---

## Part 2 — Flutter Mobile App

---

### Task 7: Flutter project scaffold

**Files:**
- Create: `app/` Flutter project
- Create: `app/pubspec.yaml`
- Modify: `app/android/app/src/main/AndroidManifest.xml`
- Modify: `app/android/app/build.gradle`

- [ ] **Step 1: Create Flutter project**

```bash
flutter create --org com.neuroride --project-name neuroride app
```

- [ ] **Step 2: Replace `app/pubspec.yaml` with full content**

```yaml
name: neuroride
description: EEG-based rider drowsiness detection
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  flutter_bluetooth_serial: ^0.4.0
  tflite_flutter: ^0.10.4
  audioplayers: ^5.2.1
  vibration: ^1.8.4
  provider: ^6.1.2
  fl_chart: ^0.66.2
  permission_handler: ^11.3.1
  hive_flutter: ^1.1.0

flutter:
  uses-material-design: true
  assets:
    - assets/neuroride_model.tflite
    - assets/scaler_params.json
    - assets/alert_sound.mp3
```

- [ ] **Step 3: Create assets folder and add alert sound**

```bash
mkdir app/assets
```

Download a free short alert beep MP3 (e.g. from freesound.org) and save as `app/assets/alert_sound.mp3`. Any short MP3 works — it is the alarm sound played when drowsiness is detected.

If you have no MP3 handy, create an empty placeholder so the build succeeds, then replace it later:
```bash
# PowerShell one-liner to create a tiny valid MP3 placeholder:
# (AudioPlayers will fail silently if the file is invalid — vibration still works)
echo "" > app/assets/alert_sound.mp3
```

- [ ] **Step 4: Replace `app/android/app/src/main/AndroidManifest.xml`**

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.BLUETOOTH"/>
    <uses-permission android:name="android.permission.BLUETOOTH_ADMIN"/>
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT"/>
    <uses-permission android:name="android.permission.BLUETOOTH_SCAN"/>
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION"/>
    <uses-permission android:name="android.permission.VIBRATE"/>
    <uses-permission android:name="android.permission.INTERNET"/>

    <application
        android:label="NeuroRide"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:taskAffinity=""
            android:theme="@style/LaunchTheme"
            android:configChanges="orientation|keyboardHidden|keyboard|screenSize|smallestScreenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
            android:hardwareAccelerated="true"
            android:windowSoftInputMode="adjustResize">
            <meta-data
                android:name="io.flutter.embedding.android.NormalTheme"
                android:resource="@style/NormalTheme"/>
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        <meta-data
            android:name="flutterEmbedding"
            android:value="2"/>
    </application>

    <queries>
        <intent>
            <action android:name="android.intent.action.VIEW"/>
        </intent>
    </queries>
</manifest>
```

- [ ] **Step 5: Set minSdkVersion in `app/android/app/build.gradle`**

Find the line containing `minSdkVersion` and change it to:
```gradle
minSdkVersion 21
```

- [ ] **Step 6: Get packages**

```bash
cd app && flutter pub get
```

Expected: Resolves without errors.

- [ ] **Step 7: Verify debug build**

```bash
cd app && flutter build apk --debug 2>&1 | tail -3
```

Expected: `Built build/app/outputs/flutter-apk/app-debug.apk`

- [ ] **Step 8: Commit**

```bash
cd ..
git add app/
git commit -m "feat: scaffold Flutter project with packages and permissions"
```

---

### Task 8: EegData model

**Files:**
- Create: `app/lib/models/eeg_data.dart`
- Create: `app/test/models/eeg_data_test.dart`

- [ ] **Step 1: Create `app/lib/models/eeg_data.dart`**

```dart
import 'dart:convert';
import 'package:flutter/services.dart';

class EegData {
  final double delta;
  final double theta;
  final double lowAlpha;
  final double highAlpha;
  final double lowBeta;
  final double highBeta;
  final double lowGamma;
  final double midGamma;
  final int attention;
  final int meditation;
  final int signalStrength;
  final DateTime timestamp;

  const EegData({
    required this.delta,
    required this.theta,
    required this.lowAlpha,
    required this.highAlpha,
    required this.lowBeta,
    required this.highBeta,
    required this.lowGamma,
    required this.midGamma,
    required this.attention,
    required this.meditation,
    required this.signalStrength,
    required this.timestamp,
  });

  bool get hasGoodSignal => signalStrength == 0;

  /// Normalises a raw TGAM1 24-bit band value (0–16 777 215) to 0–100.
  /// 100 000 is a representative maximum for typical indoor EEG readings.
  static double normalizeRaw(double raw) =>
      (raw / 100000.0 * 100.0).clamp(0.0, 100.0);

  /// Builds the 14-element feature vector expected by the TFLite model.
  /// Applies StandardScaler normalisation using saved scaler params.
  List<double> toFeatureVector(Map<String, dynamic> scalerParams) {
    final mean  = List<double>.from(scalerParams['mean']  as List);
    final scale = List<double>.from(scalerParams['scale'] as List);

    final thetaBeta  = theta / (lowBeta + highBeta + 0.001);
    final alphaBeta  = (lowAlpha + highAlpha) / (lowBeta + highBeta + 0.001);
    final deltaAB    = delta / (lowAlpha + highAlpha + 0.001);
    final totalPower = delta + theta + lowAlpha + highAlpha +
        lowBeta + highBeta + lowGamma + midGamma;

    final raw = [
      delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma,
      attention.toDouble(), meditation.toDouble(),
      thetaBeta, alphaBeta, deltaAB, totalPower,
    ];

    return List.generate(14, (i) => (raw[i] - mean[i]) / scale[i]);
  }
}

/// Loads and caches scaler parameters from Flutter assets.
class ScalerParams {
  static Map<String, dynamic>? _cache;

  static Future<Map<String, dynamic>> load() async {
    if (_cache != null) return _cache!;
    final raw = await rootBundle.loadString('assets/scaler_params.json');
    _cache = json.decode(raw) as Map<String, dynamic>;
    return _cache!;
  }
}
```

- [ ] **Step 2: Create `app/test/models/eeg_data_test.dart`**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:neuroride/models/eeg_data.dart';

EegData _sample({int signalStrength = 0}) => EegData(
      delta: 5.0, theta: 8.0, lowAlpha: 3.0, highAlpha: 2.0,
      lowBeta: 3.0, highBeta: 4.0, lowGamma: 2.0, midGamma: 1.0,
      attention: 48, meditation: 57,
      signalStrength: signalStrength,
      timestamp: DateTime(2026),
    );

Map<String, dynamic> _identityScaler() => {
      'mean':          List.filled(14, 0.0),
      'scale':         List.filled(14, 1.0),
      'feature_names': List.generate(14, (i) => 'f$i'),
    };

void main() {
  group('EegData.toFeatureVector', () {
    test('returns exactly 14 elements', () {
      final v = _sample().toFeatureVector(_identityScaler());
      expect(v.length, equals(14));
    });

    test('first 10 elements match raw values when scaler is identity', () {
      final v = _sample().toFeatureVector(_identityScaler());
      expect(v[0], closeTo(5.0,  0.001)); // delta
      expect(v[8], closeTo(48.0, 0.001)); // attention
      expect(v[9], closeTo(57.0, 0.001)); // meditation
    });
  });

  group('EegData.normalizeRaw', () {
    test('100 000 maps to 100', () =>
        expect(EegData.normalizeRaw(100000), closeTo(100.0, 0.001)));
    test('50 000 maps to 50',  () =>
        expect(EegData.normalizeRaw(50000),  closeTo(50.0,  0.001)));
    test('0 maps to 0',        () =>
        expect(EegData.normalizeRaw(0),      closeTo(0.0,   0.001)));
    test('values above max are clamped to 100', () =>
        expect(EegData.normalizeRaw(999999), closeTo(100.0, 0.001)));
  });

  group('EegData.hasGoodSignal', () {
    test('true when signalStrength == 0',   () =>
        expect(_sample(signalStrength: 0).hasGoodSignal,   isTrue));
    test('false when signalStrength == 200', () =>
        expect(_sample(signalStrength: 200).hasGoodSignal, isFalse));
  });
}
```

- [ ] **Step 3: Run tests**

```bash
cd app && flutter test test/models/eeg_data_test.dart -v
```

Expected: 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
cd ..
git add app/lib/models/eeg_data.dart app/test/models/eeg_data_test.dart
git commit -m "feat: add EegData model with feature vector computation"
```

---

### Task 9: ThinkGear packet parser

**Files:**
- Create: `app/lib/services/thinkgear_parser.dart`
- Create: `app/test/services/thinkgear_parser_test.dart`

- [ ] **Step 1: Create `app/lib/services/thinkgear_parser.dart`**

```dart
import 'dart:typed_data';
import '../models/eeg_data.dart';

/// Decodes NeuroSky TGAM1 ThinkGear binary packet stream.
///
/// Packet format:
///   [0xAA][0xAA][payloadLen][...payload...][checksum]
///
/// Payload codes used by MindLink:
///   0x02  signal quality  (1 byte,  0 = good contact)
///   0x04  eSense Attention (1 byte, 0–100)
///   0x05  eSense Meditation(1 byte, 0–100)
///   0x83  EEG power bands  (length byte = 24, then 8 × 3-byte big-endian uint32)
///           order: delta, theta, low-alpha, high-alpha,
///                  low-beta, high-beta, low-gamma, mid-gamma
class ThinkGearParser {
  final _buf = <int>[];

  /// Feed raw bytes from the Bluetooth stream.
  /// Returns a decoded [EegData] when a complete valid packet is found,
  /// or null if more bytes are needed.
  EegData? addBytes(Uint8List bytes) {
    _buf.addAll(bytes);
    return _tryParse();
  }

  EegData? _tryParse() {
    while (_buf.length >= 4) {
      // Locate sync header 0xAA 0xAA
      if (_buf[0] != 0xAA || _buf[1] != 0xAA) {
        _buf.removeAt(0);
        continue;
      }

      final payloadLen = _buf[2];
      if (payloadLen > 169) {         // ThinkGear spec max
        _buf.removeAt(0);
        continue;
      }

      // Full packet = 2 sync + 1 len + payload + 1 checksum
      final totalLen = 4 + payloadLen;
      if (_buf.length < totalLen) break; // wait for more bytes

      // Verify checksum: ones-complement of sum of payload bytes
      int sum = 0;
      for (var i = 3; i < 3 + payloadLen; i++) sum += _buf[i];
      final expected = (~sum) & 0xFF;

      if (expected != _buf[3 + payloadLen]) {
        _buf.removeAt(0);             // bad checksum, slide window
        continue;
      }

      final payload = _buf.sublist(3, 3 + payloadLen);
      _buf.removeRange(0, totalLen);

      final result = _decodePayload(payload);
      if (result != null) return result;
    }
    return null;
  }

  EegData? _decodePayload(List<int> payload) {
    var i = 0;
    var signalStrength = 0;
    var attention = 0;
    var meditation = 0;
    double delta = 0, theta = 0, lowAlpha = 0, highAlpha = 0;
    double lowBeta = 0, highBeta = 0, lowGamma = 0, midGamma = 0;
    var hasWaves = false;

    while (i < payload.length) {
      final code = payload[i++];

      if (code == 0x02) {
        if (i < payload.length) signalStrength = payload[i++];
      } else if (code == 0x04) {
        if (i < payload.length) attention = payload[i++];
      } else if (code == 0x05) {
        if (i < payload.length) meditation = payload[i++];
      } else if (code == 0x83) {
        if (i >= payload.length) break;
        final len = payload[i++];
        if (len == 24 && i + 24 <= payload.length) {
          delta     = _u24(payload, i);
          theta     = _u24(payload, i + 3);
          lowAlpha  = _u24(payload, i + 6);
          highAlpha = _u24(payload, i + 9);
          lowBeta   = _u24(payload, i + 12);
          highBeta  = _u24(payload, i + 15);
          lowGamma  = _u24(payload, i + 18);
          midGamma  = _u24(payload, i + 21);
          hasWaves  = true;
          i += 24;
        } else {
          i += len;
        }
      } else if (code >= 0x80) {
        // Multi-byte extended code — skip
        if (i < payload.length) i += payload[i] + 1;
      } else {
        // Single-byte value — skip
        if (i < payload.length) i++;
      }
    }

    if (!hasWaves) return null;

    return EegData(
      delta:         EegData.normalizeRaw(delta),
      theta:         EegData.normalizeRaw(theta),
      lowAlpha:      EegData.normalizeRaw(lowAlpha),
      highAlpha:     EegData.normalizeRaw(highAlpha),
      lowBeta:       EegData.normalizeRaw(lowBeta),
      highBeta:      EegData.normalizeRaw(highBeta),
      lowGamma:      EegData.normalizeRaw(lowGamma),
      midGamma:      EegData.normalizeRaw(midGamma),
      attention:     attention,
      meditation:    meditation,
      signalStrength: signalStrength,
      timestamp:     DateTime.now(),
    );
  }

  double _u24(List<int> b, int o) =>
      ((b[o] << 16) | (b[o + 1] << 8) | b[o + 2]).toDouble();

  void reset() => _buf.clear();
}
```

- [ ] **Step 2: Create `app/test/services/thinkgear_parser_test.dart`**

```dart
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:neuroride/services/thinkgear_parser.dart';

/// Builds a syntactically correct ThinkGear packet.
Uint8List buildPacket({
  int signal = 0,
  int attention = 60,
  int meditation = 50,
  List<int>? bands, // 24 raw bytes for 8 band values (3 bytes each)
}) {
  final payload = <int>[
    0x02, signal,
    0x04, attention,
    0x05, meditation,
    0x83, 24,
    ...(bands ?? List.filled(24, 0)),
  ];
  int cs = payload.fold(0, (a, b) => a + b);
  cs = (~cs) & 0xFF;
  return Uint8List.fromList([0xAA, 0xAA, payload.length, ...payload, cs]);
}

/// Encodes an integer into 3 big-endian bytes.
List<int> u24(int value) => [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF];

void main() {
  group('ThinkGearParser', () {
    test('parses valid packet — attention and meditation', () {
      final p = ThinkGearParser();
      final result = p.addBytes(buildPacket(attention: 75, meditation: 40));
      expect(result, isNotNull);
      expect(result!.attention,   equals(75));
      expect(result.meditation,   equals(40));
      expect(result.signalStrength, equals(0));
    });

    test('returns null for incomplete packet', () {
      final p = ThinkGearParser();
      final full = buildPacket();
      final result = p.addBytes(Uint8List.fromList(full.sublist(0, 5)));
      expect(result, isNull);
    });

    test('completes after receiving second chunk', () {
      final p = ThinkGearParser();
      final full = buildPacket(attention: 55);
      p.addBytes(Uint8List.fromList(full.sublist(0, 5)));
      final result = p.addBytes(Uint8List.fromList(full.sublist(5)));
      expect(result, isNotNull);
      expect(result!.attention, equals(55));
    });

    test('rejects packet with bad checksum', () {
      final p = ThinkGearParser();
      final corrupt = buildPacket().toList()
        ..[buildPacket().length - 1] = 0x00; // zero the checksum
      final result = p.addBytes(Uint8List.fromList(corrupt));
      expect(result, isNull);
    });

    test('decodes delta = 100 000 → 100.0 after normalization', () {
      final bands = [...u24(100000), ...List.filled(21, 0)];
      final result = ThinkGearParser().addBytes(buildPacket(bands: bands));
      expect(result, isNotNull);
      expect(result!.delta, closeTo(100.0, 0.01));
    });

    test('decodes theta = 50 000 → 50.0 after normalization', () {
      final bands = [...u24(0), ...u24(50000), ...List.filled(18, 0)];
      final result = ThinkGearParser().addBytes(buildPacket(bands: bands));
      expect(result, isNotNull);
      expect(result!.theta, closeTo(50.0, 0.01));
    });

    test('reset clears internal buffer', () {
      final p = ThinkGearParser();
      final full = buildPacket(attention: 88);
      p.addBytes(Uint8List.fromList(full.sublist(0, 5))); // partial
      p.reset();
      // After reset the remainder of the original packet is meaningless
      final result = p.addBytes(Uint8List.fromList(full.sublist(5)));
      expect(result, isNull); // no valid packet without the header
    });
  });
}
```

- [ ] **Step 3: Run tests**

```bash
cd app && flutter test test/services/thinkgear_parser_test.dart -v
```

Expected: 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
cd ..
git add app/lib/services/thinkgear_parser.dart app/test/services/thinkgear_parser_test.dart
git commit -m "feat: add ThinkGear TGAM1 binary parser with tests"
```

---

### Task 10: BluetoothService

**Files:**
- Create: `app/lib/services/bluetooth_service.dart`

- [ ] **Step 1: Create `app/lib/services/bluetooth_service.dart`**

```dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

enum BtState { disconnected, connecting, connected, error }

class BluetoothService {
  BluetoothConnection? _conn;

  final _stateCtrl = StreamController<BtState>.broadcast();
  final _dataCtrl  = StreamController<Uint8List>.broadcast();

  Stream<BtState>    get stateStream => _stateCtrl.stream;
  Stream<Uint8List>  get dataStream  => _dataCtrl.stream;
  bool               get isConnected => _conn?.isConnected ?? false;

  /// Returns paired Bluetooth Classic devices from Android settings.
  Future<List<BluetoothDevice>> getPairedDevices() =>
      FlutterBluetoothSerial.instance.getBondedDevices();

  /// Connects to [address] (MAC string, e.g. "00:81:F9:XX:XX:XX").
  Future<void> connect(String address) async {
    _stateCtrl.add(BtState.connecting);
    try {
      _conn = await BluetoothConnection.toAddress(address)
          .timeout(const Duration(seconds: 12));
      _stateCtrl.add(BtState.connected);

      _conn!.input!.listen(
        (Uint8List data) => _dataCtrl.add(data),
        onDone:  () => _stateCtrl.add(BtState.disconnected),
        onError: (_) => _stateCtrl.add(BtState.error),
      );
    } catch (_) {
      _stateCtrl.add(BtState.error);
      rethrow;
    }
  }

  Future<void> disconnect() async {
    await _conn?.close();
    _conn = null;
    _stateCtrl.add(BtState.disconnected);
  }

  void dispose() {
    disconnect();
    _stateCtrl.close();
    _dataCtrl.close();
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/services/bluetooth_service.dart
git commit -m "feat: add BluetoothService Bluetooth Classic SPP"
```

---

### Task 11: TfliteService

**Files:**
- Create: `app/lib/services/tflite_service.dart`

- [ ] **Step 1: Create `app/lib/services/tflite_service.dart`**

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/eeg_data.dart';

const double _safeThreshold   = 0.35;
const double _dangerThreshold = 0.50;

enum DrowsinessLevel { safe, warning, danger }

class PredictionResult {
  final double        probability;
  final DrowsinessLevel level;

  const PredictionResult({required this.probability, required this.level});

  bool   get isDanger => level == DrowsinessLevel.danger;

  String get label {
    switch (level) {
      case DrowsinessLevel.safe:    return '✅ SAFE';
      case DrowsinessLevel.warning: return '⚠️ WARNING';
      case DrowsinessLevel.danger:  return '🚨 DANGER';
    }
  }
}

class TfliteService {
  Interpreter? _interp;
  bool get isLoaded => _interp != null;

  Future<void> load() async {
    _interp = await Interpreter.fromAsset('assets/neuroride_model.tflite');
  }

  /// Runs inference on one EEG reading.
  /// Returns [PredictionResult.safe] if model not yet loaded.
  PredictionResult predict(EegData data, Map<String, dynamic> scalerParams) {
    if (_interp == null) {
      return const PredictionResult(
          probability: 0, level: DrowsinessLevel.safe);
    }

    final features = data.toFeatureVector(scalerParams);
    // TFLite expects [[f0, f1, ..., f13]] (batch of 1)
    final input  = [features];
    final output = [[0.0]];

    _interp!.run(input, output);

    final prob = (output[0][0] as double).clamp(0.0, 1.0);
    final DrowsinessLevel lvl;
    if (prob < _safeThreshold) {
      lvl = DrowsinessLevel.safe;
    } else if (prob < _dangerThreshold) {
      lvl = DrowsinessLevel.warning;
    } else {
      lvl = DrowsinessLevel.danger;
    }
    return PredictionResult(probability: prob, level: lvl);
  }

  void dispose() {
    _interp?.close();
    _interp = null;
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/services/tflite_service.dart
git commit -m "feat: add TfliteService on-device inference"
```

---

### Task 12: AlertService

**Files:**
- Create: `app/lib/services/alert_service.dart`

- [ ] **Step 1: Create `app/lib/services/alert_service.dart`**

```dart
import 'package:audioplayers/audioplayers.dart';
import 'package:vibration/vibration.dart';

class AlertService {
  final _player = AudioPlayer();
  bool _active = false;
  bool get isActive => _active;

  Future<void> triggerAlarm() async {
    if (_active) return;
    _active = true;

    // Vibrate: three strong pulses
    if (await Vibration.hasVibrator() ?? false) {
      Vibration.vibrate(
        pattern:     [0, 600, 200, 600, 200, 600],
        intensities: [0, 255,   0, 255,   0, 255],
      );
    }

    // Play alert sound (fails silently if file is missing/invalid)
    try {
      await _player.play(AssetSource('alert_sound.mp3'));
    } catch (_) {}
  }

  Future<void> stopAlarm() async {
    _active = false;
    await _player.stop();
    Vibration.cancel();
  }

  void dispose() {
    _player.dispose();
    Vibration.cancel();
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/services/alert_service.dart
git commit -m "feat: add AlertService audio + vibration"
```

---

### Task 13: EegProvider

**Files:**
- Create: `app/lib/providers/eeg_provider.dart`

- [ ] **Step 1: Create `app/lib/providers/eeg_provider.dart`**

```dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import '../models/eeg_data.dart';
import '../services/bluetooth_service.dart';
import '../services/thinkgear_parser.dart';
import '../services/tflite_service.dart';
import '../services/alert_service.dart';

const int _alarmThreshold = 3;   // consecutive danger readings before alarm
const int _maxHistory     = 60;  // data points kept in chart history

class EegProvider extends ChangeNotifier {
  final BluetoothService _bt;
  final ThinkGearParser  _parser;
  final TfliteService    _tflite;
  final AlertService     _alerts;

  Map<String, dynamic>? _scalerParams;

  StreamSubscription<Uint8List>? _dataSub;
  StreamSubscription<BtState>?   _stateSub;

  EegData?          _lastReading;
  PredictionResult? _lastPrediction;
  BtState           _btState = BtState.disconnected;
  int               _consecutiveDanger = 0;
  bool              _alarmActive = false;

  final _attentionHistory   = <double>[];
  final _meditationHistory  = <double>[];
  final _probabilityHistory = <double>[];

  EegProvider({
    required BluetoothService bluetooth,
    required ThinkGearParser  parser,
    required TfliteService    tflite,
    required AlertService     alertService,
  })  : _bt      = bluetooth,
        _parser  = parser,
        _tflite  = tflite,
        _alerts  = alertService;

  // ── Public getters ──────────────────────────────────────────────────────────
  EegData?          get lastReading        => _lastReading;
  PredictionResult? get lastPrediction     => _lastPrediction;
  BtState           get connectionState    => _btState;
  bool              get isConnected        => _btState == BtState.connected;
  bool              get alarmActive        => _alarmActive;
  List<double>      get attentionHistory   => List.unmodifiable(_attentionHistory);
  List<double>      get meditationHistory  => List.unmodifiable(_meditationHistory);
  List<double>      get probabilityHistory => List.unmodifiable(_probabilityHistory);

  // ── Initialisation ──────────────────────────────────────────────────────────
  Future<void> initialize(Map<String, dynamic> scalerParams) async {
    _scalerParams = scalerParams;
    await _tflite.load();

    _stateSub = _bt.stateStream.listen((s) {
      _btState = s;
      notifyListeners();
    });
  }

  // ── Bluetooth ───────────────────────────────────────────────────────────────
  Future<List<BluetoothDevice>> getPairedDevices() => _bt.getPairedDevices();

  Future<void> connect(String address) async {
    await _bt.connect(address);
    _dataSub = _bt.dataStream.listen(_onData);
  }

  Future<void> disconnect() async {
    await _dataSub?.cancel();
    await _bt.disconnect();
    _parser.reset();
  }

  // ── Data processing ─────────────────────────────────────────────────────────
  void _onData(Uint8List bytes) {
    final reading = _parser.addBytes(bytes);
    if (reading == null) return;
    _lastReading = reading;

    if (!reading.hasGoodSignal || _scalerParams == null || !_tflite.isLoaded) {
      notifyListeners();
      return;
    }

    final pred = _tflite.predict(reading, _scalerParams!);
    _lastPrediction = pred;

    _push(_attentionHistory,   reading.attention.toDouble());
    _push(_meditationHistory,  reading.meditation.toDouble());
    _push(_probabilityHistory, pred.probability);

    if (pred.isDanger) {
      _consecutiveDanger++;
      if (_consecutiveDanger >= _alarmThreshold && !_alarmActive) {
        _alarmActive = true;
        _alerts.triggerAlarm();
      }
    } else {
      _consecutiveDanger = 0;
    }

    notifyListeners();
  }

  void _push(List<double> list, double value) {
    list.add(value);
    if (list.length > _maxHistory) list.removeAt(0);
  }

  // ── Alarm control ────────────────────────────────────────────────────────────
  Future<void> dismissAlarm() async {
    _alarmActive       = false;
    _consecutiveDanger = 0;
    await _alerts.stopAlarm();
    notifyListeners();
  }

  @override
  void dispose() {
    _dataSub?.cancel();
    _stateSub?.cancel();
    _bt.dispose();
    _tflite.dispose();
    _alerts.dispose();
    super.dispose();
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/providers/eeg_provider.dart
git commit -m "feat: add EegProvider with 3-reading alarm logic"
```

---

### Task 14: SplashScreen

**Files:**
- Create: `app/lib/screens/splash_screen.dart`

- [ ] **Step 1: Create `app/lib/screens/splash_screen.dart`**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:permission_handler/permission_handler.dart';
import 'device_scan_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  String _status = 'Requesting permissions…';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _requestPermissions();
    if (!mounted) return;
    await _enableBluetooth();
    if (!mounted) return;
    setState(() => _status = 'Ready!');
    await Future.delayed(const Duration(milliseconds: 400));
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const DeviceScanScreen()),
    );
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.bluetoothConnect,
      Permission.bluetoothScan,
      Permission.location,
    ].request();
  }

  Future<void> _enableBluetooth() async {
    setState(() => _status = 'Checking Bluetooth…');
    final state = await FlutterBluetoothSerial.instance.state;
    if (state == BluetoothState.STATE_OFF) {
      setState(() => _status = 'Enabling Bluetooth…');
      await FlutterBluetoothSerial.instance.requestEnable();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.memory, size: 80, color: Color(0xFF00E5FF)),
            const SizedBox(height: 20),
            const Text('NeuroRide',
                style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold,
                    color: Colors.white, letterSpacing: 2)),
            const SizedBox(height: 6),
            const Text('EEG Rider Safety',
                style: TextStyle(fontSize: 15, color: Colors.white54)),
            const SizedBox(height: 48),
            const CircularProgressIndicator(color: Color(0xFF00E5FF)),
            const SizedBox(height: 20),
            Text(_status,
                style: const TextStyle(color: Colors.white60, fontSize: 14)),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/screens/splash_screen.dart
git commit -m "feat: add SplashScreen with BT + permission setup"
```

---

### Task 15: DeviceScanScreen

**Files:**
- Create: `app/lib/screens/device_scan_screen.dart`

- [ ] **Step 1: Create `app/lib/screens/device_scan_screen.dart`**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:provider/provider.dart';
import '../providers/eeg_provider.dart';
import 'dashboard_screen.dart';

class DeviceScanScreen extends StatefulWidget {
  const DeviceScanScreen({super.key});
  @override
  State<DeviceScanScreen> createState() => _DeviceScanScreenState();
}

class _DeviceScanScreenState extends State<DeviceScanScreen> {
  List<BluetoothDevice> _devices = [];
  bool   _loading      = true;
  String? _connectingTo;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final devices = await context.read<EegProvider>().getPairedDevices();
    setState(() { _devices = devices; _loading = false; });
  }

  Future<void> _connect(BluetoothDevice device) async {
    setState(() => _connectingTo = device.address);
    try {
      await context.read<EegProvider>().connect(device.address);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const DashboardScreen()),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _connectingTo = null);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Connection failed: $e'),
        backgroundColor: Colors.redAccent,
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: const Text('Select MindLink Headband',
            style: TextStyle(color: Colors.white)),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white70),
            onPressed: () { setState(() => _loading = true); _load(); },
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF00E5FF)))
          : _devices.isEmpty
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(28),
                    child: Text(
                      'No paired devices found.\n\n'
                      'Pair your FT&S MindLink headband in Android Settings → '
                      'Connected devices → Pair new device, then return here.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white54, fontSize: 15, height: 1.5),
                    ),
                  ),
                )
              : ListView.builder(
                  itemCount: _devices.length,
                  itemBuilder: (_, i) {
                    final d = _devices[i];
                    final connecting = _connectingTo == d.address;
                    final isMindLink = (d.name ?? '').toLowerCase().contains('mindlink') ||
                        (d.name ?? '').toLowerCase().contains('neurosky');
                    return ListTile(
                      leading: Icon(Icons.bluetooth,
                          color: isMindLink ? const Color(0xFF00E5FF) : Colors.white38),
                      title: Text(d.name ?? 'Unknown',
                          style: TextStyle(
                              color: isMindLink ? Colors.white : Colors.white70,
                              fontWeight: isMindLink ? FontWeight.bold : FontWeight.normal)),
                      subtitle: Text(d.address,
                          style: const TextStyle(color: Colors.white30, fontSize: 11)),
                      trailing: connecting
                          ? const SizedBox(width: 20, height: 20,
                              child: CircularProgressIndicator(
                                  strokeWidth: 2, color: Color(0xFF00E5FF)))
                          : const Icon(Icons.chevron_right, color: Colors.white38),
                      onTap: connecting ? null : () => _connect(d),
                    );
                  },
                ),
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/screens/device_scan_screen.dart
git commit -m "feat: add DeviceScanScreen paired device list"
```

---

### Task 16: DashboardScreen

**Files:**
- Create: `app/lib/screens/dashboard_screen.dart`

- [ ] **Step 1: Create `app/lib/screens/dashboard_screen.dart`**

```dart
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/eeg_provider.dart';
import '../services/bluetooth_service.dart';
import '../services/tflite_service.dart';
import 'device_scan_screen.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<EegProvider>(
      builder: (context, prov, _) => Scaffold(
        backgroundColor: const Color(0xFF0D1117),
        appBar: _buildAppBar(context, prov),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (prov.alarmActive) _AlarmBanner(provider: prov),
              if (prov.alarmActive) const SizedBox(height: 12),
              _DrowsinessGauge(provider: prov),
              const SizedBox(height: 12),
              _SignalRow(provider: prov),
              const SizedBox(height: 12),
              _AttentionChart(provider: prov),
              const SizedBox(height: 12),
              _EegValuesCard(provider: prov),
            ],
          ),
        ),
      ),
    );
  }

  AppBar _buildAppBar(BuildContext ctx, EegProvider prov) => AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: Row(children: [
          Icon(Icons.circle, size: 10,
              color: prov.isConnected ? Colors.greenAccent : Colors.redAccent),
          const SizedBox(width: 8),
          Text(prov.isConnected ? 'NeuroRide — Live' : 'Disconnected',
              style: const TextStyle(color: Colors.white, fontSize: 16)),
        ]),
        actions: [
          IconButton(
            icon: const Icon(Icons.bluetooth_disabled, color: Colors.white54),
            onPressed: () async {
              await prov.disconnect();
              if (!ctx.mounted) return;
              Navigator.of(ctx).pushReplacement(
                  MaterialPageRoute(builder: (_) => const DeviceScanScreen()));
            },
          ),
        ],
      );
}

// ── Alarm banner ──────────────────────────────────────────────────────────────
class _AlarmBanner extends StatelessWidget {
  final EegProvider provider;
  const _AlarmBanner({required this.provider});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: provider.dismissAlarm,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: const Color(0xFF7B1C1C),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.redAccent, width: 2),
          ),
          child: const Row(children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orangeAccent, size: 32),
            SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('DROWSINESS DETECTED',
                      style: TextStyle(color: Colors.white,
                          fontWeight: FontWeight.bold, fontSize: 17)),
                  Text('Tap to dismiss  •  Pull over safely',
                      style: TextStyle(color: Colors.white70, fontSize: 12)),
                ],
              ),
            ),
          ]),
        ),
      );
}

// ── Drowsiness gauge ──────────────────────────────────────────────────────────
class _DrowsinessGauge extends StatelessWidget {
  final EegProvider provider;
  const _DrowsinessGauge({required this.provider});

  @override
  Widget build(BuildContext context) {
    final pred = provider.lastPrediction;
    final prob = pred?.probability ?? 0.0;
    final label = pred?.label ?? '— Waiting —';
    final color = prob >= 0.50
        ? Colors.redAccent
        : prob >= 0.35
            ? Colors.orangeAccent
            : Colors.greenAccent;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          color: const Color(0xFF161B22),
          borderRadius: BorderRadius.circular(16)),
      child: Column(children: [
        Text(label,
            style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold,
                color: color)),
        const SizedBox(height: 14),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: prob, minHeight: 14,
            backgroundColor: Colors.white10,
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
        const SizedBox(height: 8),
        Text('${(prob * 100).toStringAsFixed(1)}% drowsy probability',
            style: const TextStyle(color: Colors.white38, fontSize: 12)),
      ]),
    );
  }
}

// ── Signal quality row ────────────────────────────────────────────────────────
class _SignalRow extends StatelessWidget {
  final EegProvider provider;
  const _SignalRow({required this.provider});

  @override
  Widget build(BuildContext context) {
    final r = provider.lastReading;
    if (r == null) {
      return const Center(
          child: Text('Waiting for EEG packets…',
              style: TextStyle(color: Colors.white30, fontSize: 13)));
    }
    final good = r.hasGoodSignal;
    return Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      Icon(good ? Icons.wifi : Icons.wifi_off,
          size: 16, color: good ? Colors.greenAccent : Colors.orange),
      const SizedBox(width: 6),
      Text(good ? 'Good signal' : 'Poor signal — adjust headband',
          style: TextStyle(
              color: good ? Colors.greenAccent : Colors.orange, fontSize: 13)),
    ]);
  }
}

// ── Attention / probability chart ─────────────────────────────────────────────
class _AttentionChart extends StatelessWidget {
  final EegProvider provider;
  const _AttentionChart({required this.provider});

  @override
  Widget build(BuildContext context) {
    final hist = provider.attentionHistory;
    return Container(
      height: 140,
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
      decoration: BoxDecoration(
          color: const Color(0xFF161B22),
          borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Attention (last 60 s)',
              style: TextStyle(color: Colors.white38, fontSize: 11)),
          const SizedBox(height: 6),
          Expanded(
            child: hist.isEmpty
                ? const Center(
                    child: Text('Chart will appear here',
                        style: TextStyle(color: Colors.white12, fontSize: 12)))
                : LineChart(LineChartData(
                    minY: 0, maxY: 100,
                    gridData: const FlGridData(show: false),
                    titlesData: const FlTitlesData(show: false),
                    borderData: FlBorderData(show: false),
                    lineBarsData: [
                      LineChartBarData(
                        spots: hist.asMap().entries
                            .map((e) => FlSpot(e.key.toDouble(), e.value))
                            .toList(),
                        isCurved: true,
                        color: const Color(0xFF00E5FF),
                        barWidth: 2,
                        dotData: const FlDotData(show: false),
                        belowBarData: BarAreaData(
                            show: true,
                            color: const Color(0xFF00E5FF).withOpacity(0.08)),
                      ),
                    ],
                  )),
          ),
        ],
      ),
    );
  }
}

// ── Raw EEG values card ───────────────────────────────────────────────────────
class _EegValuesCard extends StatelessWidget {
  final EegProvider provider;
  const _EegValuesCard({required this.provider});

  @override
  Widget build(BuildContext context) {
    final r = provider.lastReading;
    if (r == null) return const SizedBox.shrink();
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: const Color(0xFF161B22),
          borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Live EEG Values',
              style: TextStyle(
                  color: Colors.white54, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8, runSpacing: 8,
            children: [
              _Chip('Attention',  r.attention.toDouble(),  Colors.cyanAccent),
              _Chip('Meditation', r.meditation.toDouble(), Colors.purpleAccent),
              _Chip('Delta',      r.delta,                 Colors.blueAccent),
              _Chip('Theta',      r.theta,                 Colors.tealAccent),
              _Chip('α Low',      r.lowAlpha,              Colors.greenAccent),
              _Chip('α High',     r.highAlpha,             Colors.lightGreenAccent),
              _Chip('β Low',      r.lowBeta,               Colors.orangeAccent),
              _Chip('β High',     r.highBeta,              Colors.deepOrangeAccent),
            ],
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final double value;
  final Color  color;
  const _Chip(this.label, this.value, this.color);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withOpacity(0.35)),
        ),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(label,
              style: TextStyle(color: color.withOpacity(0.7), fontSize: 10)),
          Text(value.toStringAsFixed(1),
              style: TextStyle(color: color, fontWeight: FontWeight.bold)),
        ]),
      );
}
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add app/lib/screens/dashboard_screen.dart
git commit -m "feat: add DashboardScreen live chart + drowsiness gauge + alarm"
```

---

### Task 17: main.dart — wire everything together

**Files:**
- Modify: `app/lib/main.dart`

- [ ] **Step 1: Replace `app/lib/main.dart`**

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'models/eeg_data.dart';
import 'providers/eeg_provider.dart';
import 'screens/splash_screen.dart';
import 'services/alert_service.dart';
import 'services/bluetooth_service.dart';
import 'services/thinkgear_parser.dart';
import 'services/tflite_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const NeuroRideApp());
}

class NeuroRideApp extends StatelessWidget {
  const NeuroRideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: ScalerParams.load(),
      builder: (context, snapshot) {
        // Show loading spinner until scaler params are ready
        if (!snapshot.hasData) {
          return const MaterialApp(
            debugShowCheckedModeBanner: false,
            home: Scaffold(
              backgroundColor: Color(0xFF0D1117),
              body: Center(
                child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
              ),
            ),
          );
        }

        return ChangeNotifierProvider(
          create: (_) {
            final provider = EegProvider(
              bluetooth:    BluetoothService(),
              parser:       ThinkGearParser(),
              tflite:       TfliteService(),
              alertService: AlertService(),
            );
            // Initialize asynchronously; UI will rebuild once ready
            provider.initialize(snapshot.data!);
            return provider;
          },
          child: MaterialApp(
            title: 'NeuroRide',
            debugShowCheckedModeBanner: false,
            theme: ThemeData(
              brightness: Brightness.dark,
              colorScheme: const ColorScheme.dark(
                primary:   Color(0xFF00E5FF),
                secondary: Color(0xFF00E5FF),
              ),
              scaffoldBackgroundColor: const Color(0xFF0D1117),
              appBarTheme: const AppBarTheme(
                backgroundColor: Color(0xFF161B22),
                elevation: 0,
              ),
            ),
            home: const SplashScreen(),
          ),
        );
      },
    );
  }
}
```

- [ ] **Step 2: Run flutter analyze**

```bash
cd app && flutter analyze
```

Expected: No errors. Warnings about deprecated APIs are acceptable.

- [ ] **Step 3: Build debug APK**

```bash
cd app && flutter build apk --debug
```

Expected: `Built build/app/outputs/flutter-apk/app-debug.apk`

- [ ] **Step 4: Install on Android device (USB debug enabled)**

```bash
cd app && flutter run
```

Expected: App launches → SplashScreen requests permissions → DeviceScanScreen shows paired BT devices.

- [ ] **Step 5: Final commit**

```bash
cd ..
git add app/lib/main.dart
git commit -m "feat: wire main.dart — NeuroRide app complete"
```

---

## End-to-End Verification Checklist

- [ ] Run Python pipeline in order: `step1` → `step2` → `step3` → `step4` → `step5`
- [ ] Confirm `ml/neuroride_model.tflite` and `ml/scaler_params.json` exist
- [ ] Copy both files to `app/assets/`
- [ ] `flutter pub get` succeeds
- [ ] `flutter build apk --debug` succeeds
- [ ] Install APK on Android phone
- [ ] Pair FT&S MindLink in Android Bluetooth Settings (Settings → Connected devices → Pair new device)
- [ ] Launch app → grant all permissions → MindLink appears in device list
- [ ] Tap MindLink → Dashboard opens
- [ ] Put on headband → attention/meditation values update in real time
- [ ] Drowsiness gauge changes colour based on probability
- [ ] If 3 consecutive readings are DANGER → alarm fires (vibration + sound)
- [ ] Tap alarm banner → alarm dismisses
