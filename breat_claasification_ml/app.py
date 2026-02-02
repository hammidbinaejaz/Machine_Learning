import streamlit as st
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

st.title("Breast Cancer Prediction App")
st.write("Enter patient feature values:")

# ---- Mean Features ----
radius_mean = st.number_input("Radius Mean", value=14.0)
texture_mean = st.number_input("Texture Mean", value=20.0)
perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
area_mean = st.number_input("Area Mean", value=600.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1)
compactness_mean = st.number_input("Compactness Mean", value=0.12)
concavity_mean = st.number_input("Concavity Mean", value=0.13)
concave_points_mean = st.number_input("Concave Points Mean", value=0.09)
symmetry_mean = st.number_input("Symmetry Mean", value=0.18)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)

# ---- SE Features ----
radius_se = st.number_input("Radius SE", value=0.3)
texture_se = st.number_input("Texture SE", value=1.0)
perimeter_se = st.number_input("Perimeter SE", value=2.0)
area_se = st.number_input("Area SE", value=20.0)
smoothness_se = st.number_input("Smoothness SE", value=0.01)
compactness_se = st.number_input("Compactness SE", value=0.03)
concavity_se = st.number_input("Concavity SE", value=0.04)
concave_points_se = st.number_input("Concave Points SE", value=0.02)
symmetry_se = st.number_input("Symmetry SE", value=0.01)
fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.002)

# ---- Worst Features ----
radius_worst = st.number_input("Radius Worst", value=16.0)
texture_worst = st.number_input("Texture Worst", value=25.0)
perimeter_worst = st.number_input("Perimeter Worst", value=110.0)
area_worst = st.number_input("Area Worst", value=800.0)
smoothness_worst = st.number_input("Smoothness Worst", value=0.14)
compactness_worst = st.number_input("Compactness Worst", value=0.25)
concavity_worst = st.number_input("Concavity Worst", value=0.27)
concave_points_worst = st.number_input("Concave Points Worst", value=0.15)
symmetry_worst = st.number_input("Symmetry Worst", value=0.30)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

# Predict button
if st.button("Predict"):
    # Arrange inputs in same order as training data
    new_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                          smoothness_mean, compactness_mean, concavity_mean,
                          concave_points_mean, symmetry_mean, fractal_dimension_mean,
                          radius_se, texture_se, perimeter_se, area_se,
                          smoothness_se, compactness_se, concavity_se,
                          concave_points_se, symmetry_se, fractal_dimension_se,
                          radius_worst, texture_worst, perimeter_worst, area_worst,
                          smoothness_worst, compactness_worst, concavity_worst,
                          concave_points_worst, symmetry_worst, fractal_dimension_worst]])

    # Apply preprocessing
    new_data_scaled = scaler.transform(new_data)
    new_data_selected = selector.transform(new_data_scaled)

    # Prediction
    prediction = model.predict(new_data_selected)

    if prediction[0] == 1:
        st.error("Malignant (Cancer Detected)")
    else:
        st.success("Benign (No Cancer)")
