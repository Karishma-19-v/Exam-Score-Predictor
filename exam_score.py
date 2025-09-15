import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open("exam_model.pkl", "rb"))

st.title("Exam Score Predictor")
st.write("Enter student details to predict the exam score:")

study_hours_per_day = st.number_input("Study Hours", min_value=0.0, value=3.1, step=0.1)
attendance_percentage = st.number_input(" Attendance_percentage", min_value=0.0, value=71.5, step=0.1)
mental_health_rating = st.number_input("Mental_health_rating", min_value=0, max_value=100, value=5)
sleep_hours = st.number_input("sleep hour", min_value=0.0, max_value=100.0, value=7.5, step=1.0)

# Prediction
if st.button("Predict Score"):
    features = np.array([[study_hours_per_day, attendance_percentage, mental_health_rating, sleep_hours]])
    prediction = model.predict(features)[0]
    st.success(f" Predicted Exam Score: {prediction:.2f}")
