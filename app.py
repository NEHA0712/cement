import streamlit as st
import pandas as pd
import pickle

# Load model and feature names
model = pickle.load(open("model.pkl", "rb"))
FEATURE_NAMES = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Concrete Strength Prediction (XGBoost)", layout="centered")

st.title("🧱 Concrete Strength Prediction App (XGBoost)")

st.write("Enter concrete mixture details below to predict compressive strength (MPa).")

# Input fields
inputs = []
labels = FEATURE_NAMES  # same names used while training

for feature in labels:
    value = st.number_input(f"{feature.capitalize()}:", min_value=0.0, format="%.3f")
    inputs.append(value)

# Predict button
if st.button("Predict Strength"):
    input_df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Concrete Strength: **{prediction:.2f} MPa**")
