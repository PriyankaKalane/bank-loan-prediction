import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("loan_model.pkl", "rb"))
le = pickle.load(open("loan_encoders.pkl", "rb"))

# Page config
st.set_page_config(page_title="Bank Loan Prediction", page_icon="🏦", layout="centered")

# Title
st.title("🏦 Bank Loan Prediction System")
st.write("Fill in the applicant details below to predict loan approval.")
st.markdown("---")

# Input form — two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    education = st.selectbox("Education (Graduate?)", ["Yes", "No"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    previous_loan = st.selectbox("Previous Loan Taken", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=300000, step=10000)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)
    tenure = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    customer_bandwidth = st.selectbox("Customer Category", ["Good", "Medium", "Bad"])

st.markdown("---")

# Predict button
if st.button("Predict Loan Status", use_container_width=True):

    # Encode categorical inputs
    gender_enc = le['Gender'].transform([gender])[0]
    married_enc = le['Married'].transform([married])[0]
    education_enc = le['Education'].transform([education])[0]
    self_employed_enc = le['Self_Employed'].transform([self_employed])[0]
    previous_loan_enc = le['Previous_Loan_Taken'].transform([previous_loan])[0]
    property_area_enc = le['Property_Area'].transform([property_area])[0]
    customer_bandwidth_enc = le['Customer_Bandwith'].transform([customer_bandwidth])[0]

    # Feature order must match training
    features = np.array([[
        age,
        gender_enc,
        married_enc,
        dependents,
        education_enc,
        self_employed_enc,
        applicant_income,
        loan_amount,
        previous_loan_enc,
        cibil_score,
        property_area_enc,
        customer_bandwidth_enc,
        tenure
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        confidence = probability[1] * 100
    else:
        st.error("❌ Loan Rejected")
        confidence = probability[0] * 100

    st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
    st.caption("Prediction made using a pruned Decision Tree classifier trained on 981 loan records with 92.88% test accuracy.")
