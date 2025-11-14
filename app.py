import streamlit as st
import pandas as pd
import pickle

# Load XGBoost model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧱 Concrete Compressive Strength Prediction App (XGBoost Model)")

st.markdown("Enter concrete mixture values below:")

# User inputs
cement = st.number_input("Cement (kg/m³)", min_value=0.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0)
coarseagg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0)
fineagg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0)
age = st.number_input("Age (days)", min_value=1.0)

if st.button("Predict Strength"):
    input_df = pd.DataFrame([[
        cement, slag, flyash, water, superplasticizer,
        coarseagg, fineagg, age
    ]], columns=[
        "cement", "slag", "flyash", "water", "superplasticizer",
        "coarseagg", "fineagg", "age"
    ])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Concrete Strength: **{prediction:.2f} MPa**")
