"""Training entrypoint for the LSTM energy consumption forecaster."""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import DATA_PATH, HORIZONS, MODEL_DIR, N_FEATURES_RFE, SEQ_LEN, TARGET
from .data_prep import build_features, load_raw, resample_hourly
from .model_lstm import build_lstm, fit_model
from .rfe_select import run_rfe
from .seq_window import to_sequences, to_sequences_multi


def main(horizon_key: str = "1h", data_path: str | None = None):
    """Train the LSTM model for the requested forecast horizon."""
    data_path = data_path or DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("🚀 Loading data…")
    df = build_features(resample_hourly(load_raw(data_path)))
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    print("🔎 RFE selecting features…")
    selected, ranking = run_rfe(X, y, n_features=N_FEATURES_RFE, step=5)
    X = X[selected]

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    horizon = HORIZONS[horizon_key]
    if horizon == 1:
        seq_builder = to_sequences
        out_dim = 1
    else:
        seq_builder = to_sequences_multi
        out_dim = horizon

    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    y_train_series = pd.Series(y_train_scaled, index=y_train.index)
    y_test_series = pd.Series(y_test_scaled, index=y_test.index)

    X_train_seq, y_train_seq = seq_builder(X_train_df, y_train_series, seq_len=SEQ_LEN, horizon=horizon)
    X_test_seq, y_test_seq = seq_builder(X_test_df, y_test_series, seq_len=SEQ_LEN, horizon=horizon)

    model = build_lstm(n_features=X.shape[1], seq_len=SEQ_LEN, out_dim=out_dim)
    validation_cut = int(0.9 * len(X_train_seq))
    print("🏋️ Training LSTM…")
    fit_model(
        model,
        X_train_seq[:validation_cut],
        y_train_seq[:validation_cut],
        X_train_seq[validation_cut:],
        y_train_seq[validation_cut:],
        epochs=30,
        batch=64,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(
            {
                "seq_len": SEQ_LEN,
                "feature_order": X.columns.tolist(),
                "horizon": horizon,
                "target": TARGET,
                "selected_rfe": selected,
                "ranking": ranking,
            },
            fp,
            indent=2,
        )
    print("💾 Saved artifacts to ./models")


if __name__ == "__main__":
    main("1h")
