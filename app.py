import streamlit as st
import joblib
import numpy as np

# Load the model
joblib_file = "model/rf_model.pkl"
model = joblib.load(joblib_file)

# Title
st.title("Soil Properties Prediction App")

# Input features
st.header("Input Features")

soil_type = st.number_input("Soil Type", min_value=0, max_value=10, value=0, step=1)
diameter = st.number_input("Diameter (m)", min_value=0.0, value=1.0, step=0.01)
period = st.number_input("Period (Hours)", min_value=0.0, value=1.0, step=0.01)
effective_stress = st.number_input("Effective Stress (kPa)", min_value=0.0, value=1.0, step=0.1)
q0 = st.number_input("Q0 (kN)", min_value=0.0, value=1.0, step=0.1)

# Prepare the feature vector for prediction
features = np.array([[soil_type, diameter, period, effective_stress, q0]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Predicted value: {prediction[0]}")

# Instructions to run the app
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Input the features in the input boxes.
2. Click on the 'Predict' button to see the prediction.
""")
