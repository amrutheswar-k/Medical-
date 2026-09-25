import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("Medical Insurance Cost Predictor")
st.write("Enter individual health and lifestyle metrics below to estimate annual medical charges.")

# User Input Form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", options=["female", "male"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", options=["no", "yes"])
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    submit = st.form_submit_button("Predict Charges")

if submit:
    input_dict = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    scaled_inputs = scaler.transform(input_df)
    predicted_charge = model.predict(scaled_inputs)[0]

    final_charge = max(0.0, predicted_charge)
    st.success(f"Estimated Annual Insurance Cost: **${final_charge:,.2f}**")
