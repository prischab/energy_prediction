"""Streamlit interface for the energy consumption forecaster."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR
from src.data_prep import build_features, load_raw, resample_hourly
from src.seq_window import to_sequences, to_sequences_multi

st.set_page_config(page_title="Household Energy Forecast — RFE + LSTM", layout="centered")
st.title("⚡ Household Energy Forecast — RFE + LSTM")
st.write("Use the quick demo model or load the full LSTM artefacts to generate predictions.")

MODEL_PATH = Path(MODEL_DIR) / "demo_model.pkl"
LSTM_MODEL_PATH = Path(MODEL_DIR) / "lstm_model.keras"
SCALER_X_PATH = Path(MODEL_DIR) / "scaler_X.pkl"
SCALER_Y_PATH = Path(MODEL_DIR) / "scaler_y.pkl"
META_PATH = Path(MODEL_DIR) / "meta.json"


@st.cache_resource(show_spinner=False)
def load_demo_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_lstm_artifacts() -> Dict[str, Any]:
    """Load the TensorFlow model, scalers and meta information if present."""
    if not (LSTM_MODEL_PATH.exists() and SCALER_X_PATH.exists() and SCALER_Y_PATH.exists() and META_PATH.exists()):
        return {}

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - handled gracefully in UI
        return {"error": f"TensorFlow is required for the LSTM model ({exc})."}

    model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    with META_PATH.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return {"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y, "meta": meta}


model_demo = load_demo_model()
artifacts = load_lstm_artifacts()

status_col1, status_col2 = st.columns(2)
with status_col1:
    st.subheader("Model artefacts")
    if "error" in artifacts:
        st.error(artifacts["error"])
    elif artifacts:
        meta = artifacts["meta"]
        st.success("Loaded LSTM artefacts", icon="✅")
        st.markdown(
            "\n".join(
                [
                    f"* Horizon: **{meta['horizon']} step(s)**",
                    f"* Sequence length: **{meta['seq_len']}**",
                    f"* Features selected: **{len(meta['feature_order'])}**",
                ]
            )
        )
    else:
        st.info("LSTM artefacts not found in the `models/` directory.")

with status_col2:
    st.subheader("Demo regressor")
    if model_demo is None:
        st.warning("`models/demo_model.pkl` is missing.")
    else:
        st.success("Loaded RandomForest demo model", icon="🌲")
        st.caption("This lightweight model powers the quick manual prediction form below.")

st.divider()

# ---- Quick Demo Section ----
st.header("🔢 Quick manual prediction")
st.write("Provide the most recent readings to get an instant estimate from the demo model.")

col1, col2 = st.columns(2)
with col1:
    global_active_power = st.number_input("Global Active Power (kW)", min_value=0.0, format="%.3f")
    voltage = st.number_input("Voltage (V)", min_value=200.0, format="%.2f")
    sub_metering_1 = st.number_input("Sub Metering 1 (W-h)", min_value=0.0, format="%.2f")
    sub_metering_3 = st.number_input("Sub Metering 3 (W-h)", min_value=0.0, format="%.2f")

with col2:
    global_reactive_power = st.number_input("Global Reactive Power (kVAR)", min_value=0.0, format="%.3f")
    global_intensity = st.number_input("Global Intensity (A)", min_value=0.0, format="%.2f")
    sub_metering_2 = st.number_input("Sub Metering 2 (W-h)", min_value=0.0, format="%.2f")

if st.button("🔮 Predict with demo model"):
    features = np.array(
        [
            [
                global_active_power,
                global_reactive_power,
                voltage,
                global_intensity,
                sub_metering_1,
                sub_metering_2,
                sub_metering_3,
            ]
        ]
    )

    if model_demo is None:
        st.error("Model not found. Please ensure `models/demo_model.pkl` exists.")
    else:
        prediction = model_demo.predict(features)[0]
        st.success(f"⚡ Predicted next-hour energy consumption: **{prediction:.3f} kWh**")
else:
    st.info("Enter readings and click the button to run the demo model.")

st.divider()

# ---- LSTM Inference Section ----
st.header("🧠 Use the trained LSTM model")
st.write(
    "Upload a recent slice of the original UCI Household Power Consumption dataset. "
    "The app will build the engineered features, create the last available sequence, and return the LSTM prediction."
)

uploaded_file = st.file_uploader("Upload CSV/TXT from the UCI dataset", type=["csv", "txt"])

if uploaded_file is not None:
    if not artifacts or "error" in artifacts:
        st.error("LSTM artefacts could not be loaded. Please check the status above.")
    else:
        try:
            raw_df = load_raw(uploaded_file)
            hourly_df = resample_hourly(raw_df)
            features_df = build_features(hourly_df)
        except Exception as exc:  # pragma: no cover - user input dependent
            st.exception(exc)
        else:
            meta = artifacts["meta"]
            required_columns = meta["feature_order"]

            missing = [col for col in required_columns if col not in features_df.columns]
            if missing:
                st.error(f"Missing engineered features in uploaded data: {missing}")
            elif len(features_df) < meta["seq_len"]:
                st.error(
                    f"Need at least {meta['seq_len']} hourly rows after feature engineering; got {len(features_df)}."
                )
            else:
                features_df = features_df[required_columns + [meta["target"]]]
                X_values = artifacts["scaler_X"].transform(features_df[required_columns])
                y_values = artifacts["scaler_y"].transform(
                    features_df[meta["target"]].values.reshape(-1, 1)
                ).ravel()

                X_df = pd.DataFrame(X_values, index=features_df.index, columns=required_columns)
                y_series = pd.Series(y_values, index=features_df.index)

                if meta["horizon"] == 1:
                    X_seq, _ = to_sequences(X_df, y_series, seq_len=meta["seq_len"], horizon=meta["horizon"])
                else:
                    X_seq, _ = to_sequences_multi(
                        X_df, y_series, seq_len=meta["seq_len"], horizon=meta["horizon"]
                    )

                if len(X_seq) == 0:
                    st.error("Unable to build a sequence from the uploaded data.")
                else:
                    last_sequence = X_seq[-1:]
                    pred_scaled = artifacts["model"].predict(last_sequence)
                    pred = artifacts["scaler_y"].inverse_transform(pred_scaled)
                    if meta["horizon"] == 1:
                        st.success(
                            f"LSTM prediction for the next hour: **{float(pred.flatten()[0]):.3f} kWh**"
                        )
                    else:
                        prediction_series = pd.Series(pred.flatten())
                        st.success("Generated multi-step forecast.")
                        st.dataframe(prediction_series.to_frame("Predicted kWh"))
else:
    st.caption(
        "Tip: the dataset from https://archive.ics.uci.edu can be downloaded and a small slice uploaded here "
        "to exercise the full pipeline."
    )
