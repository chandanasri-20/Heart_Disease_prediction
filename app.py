import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")   #

st.title("🫀 Heart Disease Prediction App")
st.write("Enter patient details below:")

# Sidebar
st.sidebar.header("Patient Input")

age = st.sidebar.number_input("Age", 1, 120, 30)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)

# Convert inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.markdown("---")
st.write("Developed using Machine Learning & Streamlit")
