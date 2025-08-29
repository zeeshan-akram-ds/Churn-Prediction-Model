import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model pipeline
model_pipeline = joblib.load("voting_churn.pkl")  # Save this during training using joblib

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Binary inputs
gender = st.selectbox("Gender", ['Female', 'Male'])
senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
partner = st.selectbox("Has Partner", ['No', 'Yes'])
dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes'])
online_security = st.selectbox("Online Security", ['No', 'Yes'])
online_backup = st.selectbox("Online Backup", ['No', 'Yes'])
device_protection = st.selectbox("Device Protection", ['No', 'Yes'])
tech_support = st.selectbox("Tech Support", ['No', 'Yes'])
streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes'])
streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes'])
paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])

# One-hot encoded categorical inputs
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# Numerical inputs
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Derived feature
avg_monthly_spend = total_charges / (tenure if tenure != 0 else 1)

# Prepare input dataframe
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'PaperlessBilling': paperless_billing,
    'InternetService': internet_service,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'AvgMonthlySpend': avg_monthly_spend
}

input_df = pd.DataFrame([input_dict])

# Binary mapping
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
binary_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling'
]

for col in binary_cols:
    input_df[col] = input_df[col].map(binary_map)

# One-hot columns: ensure strings
onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
for col in onehot_cols:
    input_df[col] = input_df[col].astype(str)

if st.button("Predict"):
    prob = model_pipeline.predict_proba(input_df)[0][1]  # Probability of churn
    prediction = 1 if prob >= 0.4 else 0  # Custom threshold of 0.4

    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"**Churn Probability:** {prob:.2f} (Threshold: 0.4)")

    if prediction == 1:
        st.warning("⚠️ This customer is **likely to churn**. Consider taking retention action.")
    else:
        st.success("✅ This customer is **likely to stay**.")