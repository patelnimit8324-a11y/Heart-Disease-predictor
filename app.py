import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Health Predictor: Diabetes & Heart", layout="centered")
st.title("🩺 Health Predictor — Diabetes & Heart (Demo)")

st.markdown("""
**Important:** This is an educational demo only — not medical advice.
""")

app_mode = st.selectbox("Choose prediction type", ["Diabetes Prediction", "Heart Disease Prediction", "About / Instructions"])

if app_mode == "About / Instructions":
    st.write("""
    **How to use**
    1. Train models in Colab (or upload the model files) to create `diabetes_model.joblib`, `diabetes_scaler.joblib`, `heart_model.joblib`, `heart_scaler.joblib`.
    2. Upload those files to the same folder as this app (or modify the paths).
    3. Use the forms below to enter patient values and get a prediction.
    """)

elif app_mode == "Diabetes Prediction":
    st.header("Diabetes Prediction (Pima - demo)")
    st.write("Enter the patient's features:")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    bp = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
    skin = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, format="%.2f")
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=10.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

    if st.button("Predict Diabetes"):
        try:
            model = joblib.load('diabetes_model.joblib')
            scaler = joblib.load('diabetes_scaler.joblib')
        except Exception as e:
            st.error("Model files not found. Please ensure diabetes_model.joblib and diabetes_scaler.joblib are in the app folder.")
            st.stop()

        x = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        x_s = scaler.transform(x)
        prob = model.predict_proba(x_s)[0][1] if hasattr(model, "predict_proba") else None
        pred = model.predict(x_s)[0]
        st.write("**Predicted class:**", "Diabetic (1)" if pred==1 else "Non-diabetic (0)")
        if prob is not None:
            st.write(f"**Predicted probability (diabetes):** {prob:.2f}")

        st.info("This is just a demo. For clinical use, get validated models and medical consultation.")

elif app_mode == "Heart Disease Prediction":
    st.header("Heart Disease Prediction (UCI Cleveland - demo)")
    st.write("Enter the patient's features (a simple subset). For full model, adapt to your dataset columns.")

    # Provide a common subset (modify if your heart.csv uses different names/order)
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest pain type (cp):", [0,1,2,3])  # encoding depends on dataset
    trestbps = st.number_input("Resting blood pressure (trestbps)", value=130)
    chol = st.number_input("Serum cholesterol (chol)", value=250)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", [0,1])
    restecg = st.selectbox("Resting ECG results (restecg):", [0,1,2])
    thalach = st.number_input("Max heart rate achieved (thalach)", value=150)
    exang = st.selectbox("Exercise induced angina (exang):", [0,1])
    oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", value=1.0, format="%.2f")
    slope = st.selectbox("Slope of the peak exercise ST segment (slope):", [0,1,2])
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca):", [0,1,2,3,4])
    thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect):", [1,2,3])

    # convert sex text to numeric (if original dataset uses 1=male,0=female)
    sex_n = 1 if sex=="Male" else 0

    if st.button("Predict Heart Disease"):
        try:
            model = joblib.load('heart_model.joblib')
            scaler = joblib.load('heart_scaler.joblib')
        except Exception as e:
            st.error("Model files not found. Please ensure heart_model.joblib and heart_scaler.joblib are in the app folder.")
            st.stop()

        x = np.array([[age, sex_n, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        x_s = scaler.transform(x)
        pred = model.predict(x_s)[0]
        prob = model.predict_proba(x_s)[0][1] if hasattr(model, "predict_proba") else None

        st.write("**Predicted class:**", "Heart disease likely (1)" if pred==1 else "No heart disease (0)")
        if prob is not None:
            st.write(f"**Predicted probability (heart disease):** {prob:.2f}")

        st.info("This is a demo model. For real clinical decisions, use validated clinical tools and consult doctors.")
