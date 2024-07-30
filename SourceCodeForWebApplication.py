import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the trained model
model = load_model('heart.h5')

# Streamlit app
# st.title("BLUE MARBLE SMARTWARE")
# st.title("Heart Disease Prediction")
# st.image("icon.jpg", use_column_width=True)  # Replace with your image path
# st.title("BLUE MARBLE SMARTWARE")  # Set the page title and icon
# st.title("Heart Disease Detection Using CNN Algorithm")

col1, col2 = st.columns([1, 2])  # Adjust the column sizes as needed

# Display the image in the left column
with col1:
    st.image("icon.jpg",  use_column_width=True)  # Replace with your image path

# Display the title and other elements in the right column
with col2:
    st.title("BLUE MARBLE SMARTWARE")
    st.subheader("Heart Disease Detection Using CNN Algorithm")
# Input features
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                  format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2],
                        format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}[x])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2],
                     format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                    format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x])

# Predict button
if st.button("Predict"):
    # Create input DataFrame
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Reshape input for CNN
    input_data = np.expand_dims(input_data, axis=2)

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display result
    if predicted_class == 1:
        st.success("The model predicts: Heart Disease Present")
    else:
        st.success("The model predicts: No Heart Disease")

# Run the app with: streamlit run app.py
