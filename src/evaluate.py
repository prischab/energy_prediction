"""Evaluation helpers for the trained LSTM model."""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DATA_PATH, MODEL_DIR
from .data_prep import build_features, load_raw, resample_hourly
from .seq_window import to_sequences, to_sequences_multi


def _require_tensorflow() -> Any:
    """Import TensorFlow lazily so callers can handle the missing dependency."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency provided at runtime
        raise ImportError(
            "TensorFlow is required to load the saved LSTM model. "
            "Install it with `pip install tensorflow`."
        ) from exc

    return tf


def eval_last_test_window():
    """Evaluate the saved model on the hold-out portion of the dataset."""
    tf = _require_tensorflow()

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    df = build_features(resample_hourly(load_raw(DATA_PATH)))
    y = df[meta["target"]]
    X = df[meta["feature_order"]]

    split = int(0.8 * len(X))
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    y_test_series = pd.Series(y_test_scaled, index=y_test.index)

    if meta["horizon"] == 1:
        X_seq, y_seq = to_sequences(
            X_test_df,
            y_test_series,
            seq_len=meta["seq_len"],
            horizon=meta["horizon"],
        )
    else:
        X_seq, y_seq = to_sequences_multi(
            X_test_df,
            y_test_series,
            seq_len=meta["seq_len"],
            horizon=meta["horizon"],
        )

    preds = model.predict(X_seq, verbose=0)
    preds_inv = scaler_y.inverse_transform(preds)
    y_inv = scaler_y.inverse_transform(y_seq.reshape(-1, 1))

    mae = mean_absolute_error(y_inv.flatten(), preds_inv.flatten())
    rmse = mean_squared_error(y_inv.flatten(), preds_inv.flatten(), squared=False)
    return float(mae), float(rmse)


if __name__ == "__main__":
    print(eval_last_test_window())
