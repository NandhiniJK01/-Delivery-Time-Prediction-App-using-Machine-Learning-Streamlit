import streamlit as st
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚚 Delivery Time Prediction App")

# Input fields
distance = st.number_input("Enter the delivery distance (in km):", min_value=0.0, step=0.1)
weight = st.number_input("Enter the package weight (in kg):", min_value=0.0, step=0.1)

if st.button("Predict"):
    features = np.array([[distance, weight]])
    prediction = model.predict(features)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")
