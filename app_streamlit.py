import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Household Energy Forecast — RFE + LSTM", layout="centered")

st.title("⚡ Household Energy Forecast — RFE + LSTM")
st.write("Enter your latest household energy readings below to get a next-hour consumption prediction.")

MODEL_PATH = "models/demo_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ---- User Input Section ----
st.subheader("🔢 Input your latest readings:")

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

# ---- Predict Button ----
if st.button("🔮 Predict Energy Consumption"):
    features = np.array([[global_active_power, global_reactive_power, voltage,
                          global_intensity, sub_metering_1, sub_metering_2, sub_metering_3]])

    if model is None:
        st.error("Model not found. Please ensure 'models/demo_model.pkl' exists.")
    else:
        prediction = model.predict(features)[0]
        st.success(f"⚡ Predicted Next-Hour Energy Consumption: **{prediction:.3f} kWh**")
else:
    st.info("Enter readings and click Predict.")

st.caption("Note: This lightweight cloud version uses a scikit-learn model trained on selected RFE features.")
