import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

model  = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Diabetes Risk Prediction")
st.markdown("""
Enter the patient's diagnostic measurements below.
The model will predict the likelihood of diabetes.

""")

st.divider()

st.subheader("Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies",min_value=0, max_value=17, value=3)
    glucose     = st.slider("Glucose (mg/dL)",min_value=44, max_value=199, value=120)
    blood_pres  = st.slider("Blood Pressure (mm Hg)",min_value=24, max_value=122, value=70)
    skin_thick  = st.slider("Skin Thickness (mm)",min_value=7, max_value=99, value=23)

with col2:
    insulin     = st.slider("Insulin (μU/mL)",min_value=14, max_value=846, value=80)
    bmi         = st.slider("BMI (kg/m²)",min_value=18, max_value=67, value=32)
    dpf         = st.slider("Diabetes Pedigree Function",min_value=0.08, max_value=2.42,value=0.47, step=0.01)
    age         = st.slider("Age (years)",min_value=21, max_value=81, value=33)

st.divider()

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_pres,
                              skin_thick, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk — Diabetic (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk — Non-Diabetic (Confidence: {probability[0]*100:.1f}%)")

    st.metric("Probability of Diabetes", f"{probability[1]*100:.1f}%")
    st.progress(int(probability[1] * 100))