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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    scaler_params = {
        'mean':          scaler.mean_.tolist(),
        'scale':         scaler.scale_.tolist(),
        'feature_names': FEATURES,
    }
    with open('ml/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("Saved ml/scaler_params.json")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    weight = {0: 1.0, 1: float(n_neg / n_pos)}
    print(f"Class weight for drowsy: {weight[1]:.2f}")

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

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open('ml/neuroride_model.tflite', 'wb') as f:
        f.write(tflite_bytes)
    print(f"Saved ml/neuroride_model.tflite ({len(tflite_bytes):,} bytes)")

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    sample = X_test[:1]
    interp.set_tensor(inp[0]['index'], sample)
    interp.invoke()
    prob = float(interp.get_tensor(out[0]['index'])[0][0])
    print(f"Sanity check: prob={prob:.3f}, actual={int(y_test[0])}")
    print(f"Input shape: {inp[0]['shape']}, Output shape: {out[0]['shape']}")


if __name__ == '__main__':
    convert()
