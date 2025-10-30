import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Household Energy Forecast", layout="centered")
st.title("Household Energy Forecast — RFE style (demo)")
st.caption("Upload the last 24 engineered rows to get a prediction. This cloud version uses a lightweight model.")

MODEL_PATH = "models/demo_model.pkl"
FEATURES = [
    # put the 8–12 columns you actually used after RFE, e.g.:
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.subheader("1. Upload CSV")
file = st.file_uploader("CSV with at least 24 rows and these columns:", type=["csv"])
with st.expander("Required columns"):
    st.code("\n".join(FEATURES))

if file is not None:
    df = pd.read_csv(file)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        last_row = df[FEATURES].tail(1)
        st.write("Last row used for prediction:")
        st.dataframe(last_row)

        if model is None:
            st.warning("Model file not found on cloud. Showing dummy value.")
            st.metric("Predicted energy (kWh)", "1.234")
        else:
            y_pred = model.predict(last_row)[0]
            st.metric("Predicted energy (kWh)", f"{y_pred:.3f}")
else:
    st.info("Upload a CSV to run prediction.")
