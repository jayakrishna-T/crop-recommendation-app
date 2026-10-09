import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/crop_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.title("Crop Recommendation System")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")

temperature = st.number_input("Temperature")

humidity = st.number_input("Humidity")

ph = st.number_input("Soil pH")

rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):

    sample = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

    prediction = model.predict(sample)

    crop = encoder.inverse_transform(prediction)

    st.success(f"Recommended Crop: {crop[0]}")