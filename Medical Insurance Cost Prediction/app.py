# app.py - Streamlit GUI for Medical Insurance Cost Prediction
# ---------------------------------------------------------
# Loads the artifacts saved by application.py (train_model.py):
#   sgb_model.pkl, scaler.pkl, feature_columns.pkl,
#   continuous_cols.pkl, metrics.pkl
# and lets the user enter patient info to predict charges.
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
    layout="centered",
)

# -----------------------------------------------------------------
# Load saved artifacts (cached so it only loads once)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    continuous_cols = joblib.load("continuous_cols.pkl")
    metrics = joblib.load("metrics.pkl")
    return model, scaler, feature_columns, continuous_cols, metrics


try:
    model, scaler, feature_columns, continuous_cols, metrics = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python application.py` first "
        "to train the model and generate: sgb_model.pkl, scaler.pkl, "
        "feature_columns.pkl, continuous_cols.pkl, metrics.pkl"
    )
    st.stop()

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("💰 Medical Insurance Cost Prediction")
st.write(
    "Estimate individual medical insurance charges based on patient "
    "age, BMI, number of children, smoking habits, and region. "
    "Model: **Stochastic Gradient Boosting (SGB) Regressor**."
)

with st.expander("📊 Model performance (on 20% held-out test set)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("MAE", f"${metrics['mae']:,.0f}")
    c3.metric("MSE", f"{metrics['mse']:,.0f}")
    c4.metric("RMSE", f"${metrics['rmse']:,.0f}")

st.divider()

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
st.subheader("Patient information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=5, value=0, step=1)

with col2:
    sex = st.radio("Sex", options=["male", "female"], horizontal=True)
    smoker = st.radio("Smoker", options=["yes", "no"], horizontal=True)
    region = st.selectbox(
        "Region",
        options=["northeast", "northwest", "southeast", "southwest"],
    )

st.divider()

# -----------------------------------------------------------------
# Build a single-row dataframe matching the training pipeline exactly
# -----------------------------------------------------------------
def build_input_row(age, sex, bmi, children, smoker, region, feature_columns):
    row = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    # Reindex to guarantee the exact same column order as training
    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df


if st.button("🔮 Predict charges", type="primary", use_container_width=True):
    input_df = build_input_row(age, sex, bmi, children, smoker, region, feature_columns)

    # Apply the SAME scaler fitted during training to the SAME continuous columns
    input_scaled = input_df.copy()
    input_scaled[continuous_cols] = scaler.transform(input_df[continuous_cols])

    prediction = model.predict(input_scaled)[0]
    prediction = max(prediction, 0)  # charges can't be negative

    st.success(f"### Estimated medical insurance charges: **${prediction:,.2f}**")

    st.caption(
        "This is an estimate produced by a machine learning model trained on "
        "historical data and should not be used as the sole basis for actual "
        "premium pricing decisions."
    )

st.divider()
st.caption("Dataset: Medical Cost Personal Datasets (Kaggle) · Model: GradientBoostingRegressor (subsample=0.8)")