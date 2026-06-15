
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("cardiorisk_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ CardioRisk AI")
st.subheader("AI-Powered Heart Risk Assessment System")

st.info("Please fill all required fields (*) for the most accurate prediction.")

st.markdown("---")

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age *", 1, 120, 30)

with col2:
    sex_text = st.selectbox("Sex *", ["Male", "Female"])

sex = 1 if sex_text == "Male" else 0

st.markdown("---")

st.header("🏥 Clinical Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    trestbps = st.number_input("Resting Blood Pressure *", 50, 250, 120)

with col2:
    chol = st.number_input("Cholesterol *", 100, 600, 200)

with col3:
    thalach = st.number_input("Maximum Heart Rate *", 50, 250, 150)

st.markdown("---")

st.header("🩺 Medical Indicators")

cp_text = st.selectbox(
    "Chest Pain Type *",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_map[cp_text]

fbs_text = st.selectbox("Fasting Blood Sugar > 120 mg/dL ? *", ["No", "Yes"])
fbs = 1 if fbs_text == "Yes" else 0

exang_text = st.selectbox("Exercise Induced Angina *", ["No", "Yes"])
exang = 1 if exang_text == "Yes" else 0

restecg = st.selectbox("Resting ECG Result *", [0, 1, 2])

oldpeak = st.number_input(
    "ST Depression (oldpeak) *",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox("Slope *", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (ca) *", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia *", [0, 1, 2, 3])

st.markdown("---")

if st.button("🔍 Analyze Risk"):

    patient_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    patient_data_scaled = scaler.transform(patient_data)

    prediction = model.predict(patient_data_scaled)
    probability = model.predict_proba(patient_data_scaled)

    risk_score = probability[0][1] * 100

    st.markdown("---")
    st.subheader("📊 Assessment Result")

    st.metric("Heart Disease Risk Score", f"{risk_score:.2f}%")

    st.progress(int(risk_score))

    if risk_score < 40:
        st.success("🟢 Low Risk of Heart Disease")
        st.write("Recommendation: Maintain a healthy lifestyle and continue regular checkups.")

    elif risk_score < 70:
        st.warning("🟡 Moderate Risk of Heart Disease")
        st.write("Recommendation: Consider consulting a healthcare professional for further evaluation.")

    else:
        st.error("🔴 High Risk of Heart Disease")
        st.write("Recommendation: Please consult a healthcare professional as soon as possible.")

    st.markdown("---")

    st.subheader("📋 Patient Summary")

    st.write(f"Age: {age}")
    st.write(f"Sex: {sex_text}")
    st.write(f"Blood Pressure: {trestbps}")
    st.write(f"Cholesterol: {chol}")
    st.write(f"Maximum Heart Rate: {thalach}")

    st.markdown("---")

    st.warning(
        "Disclaimer: This tool is for educational purposes only and does not replace professional medical advice."
    )
