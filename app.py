import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("final_loan_model.pkl", "rb"))
le = pickle.load(open("final_loan_encoders.pkl", "rb"))
features = pickle.load(open("final_loan_features.pkl", "rb"))

st.set_page_config(page_title="Bank Loan Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Bank Loan Prediction System")
st.write("Fill in the applicant details below to predict loan approval.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=800000, step=10000)

with col2:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=5000000, step=100000)
    loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)

# CIBIL indicator
if cibil_score < 500:
    st.error(f"🔴 CIBIL Score {cibil_score} — High rejection risk")
elif cibil_score < 650:
    st.warning(f"🟡 CIBIL Score {cibil_score} — Medium risk")
else:
    st.success(f"🟢 CIBIL Score {cibil_score} — Good range")

st.markdown("---")

if st.button("Predict Loan Status", use_container_width=True):
    edu_enc = le['education'].transform([education])[0]
    emp_enc = le['self_employed'].transform([self_employed])[0]

    row = pd.DataFrame([[age, edu_enc, emp_enc, income_annum, loan_amount, loan_term, cibil_score]],
                       columns=features)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        confidence = probability[1] * 100
    else:
        st.error("❌ Loan Rejected")
        confidence = probability[0] * 100

    st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

    st.markdown("**Key factors:**")
    if cibil_score >= 650:
        st.write(f"• 🟢 CIBIL Score {cibil_score} — strong credit history")
    elif cibil_score >= 500:
        st.write(f"• 🟡 CIBIL Score {cibil_score} — borderline credit")
    else:
        st.write(f"• 🔴 CIBIL Score {cibil_score} — poor credit history")

    if income_annum >= loan_amount * 0.2:
        st.write(f"• 🟢 Income sufficient relative to loan amount")
    else:
        st.write(f"• 🔴 Income low relative to loan amount")

    st.caption("Model: Random Forest | Accuracy: 97.78% | Dataset: 4,269 loan records")
