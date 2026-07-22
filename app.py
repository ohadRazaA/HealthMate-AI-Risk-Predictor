import streamlit as st
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("model/healthmate_model.json")

with open("model/scaler_params.json", "r") as f:
    scaler_params = json.load(f)

scaler_mean = np.array(scaler_params["mean"])
scaler_scale = np.array(scaler_params["scale"])
feature_columns = scaler_params["columns"]  # exact training column order

st.set_page_config(page_title="HealthMate Risk Predictor")
st.title("🏥 HealthMate Risk Predictor")
st.write("Enter patient vitals to predict overall health risk.")

# Inputs

age = st.number_input("Age", min_value=1, max_value=120, value=25)

sex_label = st.selectbox("Sex", options=["Male", "Female"])
sex = 1 if sex_label == "Male" else 0  # matches training: M -> 1, F -> 0

weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=250, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=20, max_value=600, value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)

# Prediction button

if st.button("Predict Risk"):

    raw_values = {
        "age": age,
        "sex": sex,
        "systolic": systolic_bp,
        "diastolic": diastolic_bp,
        "sugar": blood_sugar,
        "weight": weight,
        "bmi": bmi
    }
    input_data = pd.DataFrame([[raw_values[col] for col in feature_columns]], columns=feature_columns)

    # CRITICAL: the model was trained on StandardScaler-transformed features,
    # so we must apply the exact same transform: (x - mean) / scale
    input_scaled = (input_data.values - scaler_mean) / scaler_scale

    prediction = model.predict(input_scaled)[0]

    risk_mapping = {
        0: {
            "level": "High Attention Required",
            "message": "Your vitals indicate a higher health risk. Consider medical consultation."
        },
        1: {
            "level": "Low Concern",
            "message": "Your vitals show relatively low risk, but continue monitoring."
        },
        2: {
            "level": "Needs Attention",
            "message": "Some indicators require monitoring and lifestyle improvements."
        },
        3: {
            "level": "Healthy Range",
            "message": "Your current vitals appear within a healthy range."
        }
    }

    result = risk_mapping[prediction]

    st.subheader(result["level"])
    st.write(result["message"])