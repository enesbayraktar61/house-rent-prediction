import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="House Rent Prediction", layout="centered")

st.title("House Rent Prediction")
st.write("Enter property details to predict the estimated monthly rent.")

# Base directory (Hugging Face repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (stored in repo root)
MODEL_PATH = os.path.join(BASE_DIR, "house_rent_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "training_columns.json")

# Load model and training columns
model = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH, "r") as f:
    training_columns = json.load(f)

st.subheader("Input Features")

# Minimal set of inputs (we fill the rest with defaults)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=800, step=10)
bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2, step=1)

city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"])
furnishing = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors/Family", "Bachelors", "Family"])
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])

# Create full input row with all required columns (defaults)
input_data = {col: 0 for col in training_columns}

# Fill known inputs (must match exact column names in the dataset)
input_data.update({
    "BHK": bhk,
    "Size": size,
    "Bathroom": bathroom,
    "City": city,
    "Furnishing Status": furnishing,
    "Tenant Preferred": tenant,
    "Area Type": area_type
})

input_df = pd.DataFrame([input_data], columns=training_columns)

if st.button("Predict Rent"):
    # Predict on log scale and convert back to original rent scale
    pred_log = model.predict(input_df)[0]
    pred_rent = np.expm1(pred_log)
    st.success(f"Estimated Monthly Rent: {pred_rent:,.0f}")

st.caption("Model: RandomForestRegressor + sklearn Pipeline (trained on log-transformed rent).")
