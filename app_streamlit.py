import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Energy Forecast (RFE + LSTM)", layout="centered")
st.title("Household Energy Forecast — RFE + LSTM")
st.caption("Upload the last 24 engineered rows with the selected features to get a next-hour prediction.")

MODELS_DIR = "models"

@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir=MODELS_DIR):
    try:
        import tensorflow as tf  # required to load the .keras model
        model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))
        sx = joblib.load(os.path.join(models_dir, "scaler_X.pkl"))
        sy = joblib.load(os.path.join(models_dir, "scaler_y.pkl"))
        meta = json.load(open(os.path.join(models_dir, "meta.json")))
        return model, sx, sy, meta, None
    except Exception as e:
        return None, None, None, None, e

# --- Diagnostics sidebar ---
with st.sidebar:
    st.header("Diagnostics")
    st.write("Working dir:", os.getcwd())
    st.write("Models dir contents:", os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else "(missing)")
    try:
        import tensorflow as tf
        st.write("TensorFlow import: OK")
    except Exception as e:
        st.write("TensorFlow import FAILED:", e)

model, scaler_X, scaler_y, meta, load_err = load_artifacts()

if load_err:
    st.error("Could not load model artifacts. You can still upload a CSV to see column checks below.")
    st.exception(load_err)

# Read meta (fallbacks if meta missing)
seq_len = int(meta.get("seq_len", 24)) if isinstance(meta, dict) else 24
features = meta.get("feature_order", []) if isinstance(meta, dict) else []
horizon = int(meta.get("horizon", 1)) if isinstance(meta, dict) else 1

with st.expander("Show required feature columns"):
    st.code("\n".join(features) if features else "(meta.json missing or unreadable)", language="text")

uploaded = st.file_uploader("Upload CSV (last 24 rows will be used)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if not features:
            st.warning("No feature list available from meta.json; cannot validate columns.")
        else:
            missing = [c for c in features if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                window = df[features].tail(seq_len).values
                if window.shape[0] < seq_len:
                    st.error(f"Need at least {seq_len} rows; got {window.shape[0]}")
                else:
                    if model is None or scaler_X is None or scaler_y is None:
                        st.warning("Model not loaded. Deploy on Streamlit Cloud or fix local TF to run predictions.")
                    else:
                        Xs = scaler_X.transform(window)
                        Xs = np.expand_dims(Xs, axis=0)
                        pred = model.predict(Xs, verbose=0)
                        pred_inv = scaler_y.inverse_transform(pred)
                        if horizon == 1:
                            st.metric("Next hour forecast (kW)", f"{pred_inv[0,0]:.3f}")
                        else:
                            out = pd.DataFrame({f"t+{i+1}": [float(pred_inv[0, i])] for i in range(pred_inv.shape[1])})
                            st.write("### Forecast Horizon")
                            st.dataframe(out)
    except Exception as e:
        st.error("Failed to parse CSV or run prediction.")
        st.exception(e)

st.markdown("---")
st.caption("Tip: If local TensorFlow fails on macOS, deploy this app on Streamlit Community Cloud.")
