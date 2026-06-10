# 🏦 Bank Loan Prediction System

An end-to-end machine learning web application that predicts whether a loan application will be approved or rejected, based on applicant credit and financial data. Built with Python and deployed via Streamlit.

---

## 🔗 Links
- **Live Demo:** *(add Streamlit URL here after deployment)*
- **GitHub:** [github.com/PriyankaKalane/bank-loan-prediction](https://github.com/PriyankaKalane/bank-loan-prediction)

---

## Problem Statement

Manual loan assessment is time-consuming, inconsistent, and prone to bias. This project builds a classification model that analyzes applicant credit scores, income, and financial profile to predict loan approval — helping banks make faster, data-driven lending decisions.

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **97.78%** |
| Rejected Precision | 98% |
| Approved Precision | 98% |
| Dataset Size | 4,269 records |
| Features Used | 7 |

---

## Key Insight

**CIBIL Score is the strongest predictor (83% feature importance):**

| CIBIL Range | Risk Level | Likely Outcome |
|---|---|---|
| 300 – 499 | 🔴 High Risk | Rejected |
| 500 – 649 | 🟡 Medium Risk | Borderline |
| 650 – 900 | 🟢 Low Risk | Approved |

Average CIBIL of approved applicants: **703**
Average CIBIL of rejected applicants: **429**

---

## Features Used

| Feature | Description |
|---|---|
| Age | Applicant age |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| Annual Income (₹) | Yearly income |
| Loan Amount (₹) | Requested loan amount |
| Loan Term (months) | Repayment duration |
| CIBIL Score | Credit score (300–900) |

---

## App Preview

> Run locally using the steps below or view the live demo link above.

---

## Methodology

1. **Data Loading** — 4,269 loan application records, 13 features
2. **Data Cleaning** — Stripped whitespace, checked for missing values, verified class distribution
3. **Feature Selection** — Selected 7 most relevant features; removed correlated asset columns
4. **Encoding** — Label encoded Education and Self Employed columns
5. **Train/Test Split** — 80% training, 20% test (random_state=42)
6. **Model Training** — Random Forest Classifier with balanced class weights
7. **Evaluation** — Classification report on test set, feature importance analysis
8. **Deployment** — Interactive Streamlit app with live CIBIL risk indicator

---

## Project Structure

```
bank-loan-prediction/
│
├── app.py                        # Streamlit web application
├── final_loan_model.pkl          # Trained Random Forest model
├── final_loan_encoders.pkl       # Label encoders for categorical features
├── final_loan_features.pkl       # Feature order for prediction
├── requirements.txt              # Python dependencies
├── bank_loan_prediction.ipynb    # Full ML pipeline notebook
└── README.md                     # Project documentation
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/PriyankaKalane/bank-loan-prediction.git
cd bank-loan-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Model training & evaluation |
| Streamlit | Web app deployment |
| Git & GitHub | Version control |

---

## About

Built as part of a Post Graduate Program in Data Science & Analytics at Imarticus Learning, Pune.

**Author:** Priyanka Kalane
**LinkedIn:** [linkedin.com/in/priyankakalane](https://linkedin.com/in/priyankakalane)
**GitHub:** [github.com/PriyankaKalane](https://github.com/PriyankaKalane)
