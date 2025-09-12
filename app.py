# app.py
#
# Streamlit app: Telco churn prediction + explainability

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.pipeline import Pipeline

# ---------- CONFIGURATION ----------
MODEL_PATH = "final_churn_model.pkl"
IMG_PR_AUC = "pr_curve.png"
IMG_ROC_AUC = "rocauc_curve.png"
IMG_SHAP_SUMMARY = "shap_summary_plot.png"
IMG_CONF_MATRIX = "confusion_matrix.png"

X_train = joblib.load("X_train.pkl")  # Sample of training data for SHAP background
# Default threshold for prediction output
DEFAULT_THRESHOLD = 0.28

# Ordered list of features
FEATURES_ORDERED = ['paymentmethod', 'internetservice', 'contract', 'seniorcitizen', 'partner', 'dependents', 'paperlessbilling', 'has_internet', 'is_new_customer', 'is_m2m', 'has_streaming', 'has_security', 'is_fiber', 'tenure', 'monthlycharges', 'NumSecurityServices', 'NumStreamingServices']

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telco Customer Churn — Prediction & Explanation")

# ---------- HELPERS ----------
@st.cache_resource
def load_model(path):
    """Loads the model pipeline."""
    return joblib.load(path)

def safe_predict_proba(model, X):
    """Returns the probability for the positive class."""
    probs = model.predict_proba(X)
    return probs[:, 1]

# ---------- LOAD MODEL ----------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from {MODEL_PATH}: {e}")
        st.stop()

# ---------- SIDEBAR NAVIGATION ----------
page = st.sidebar.radio("Navigate", ["Overview", "Prediction & Explanation", "Business Impact", "Download & About"])

# ---------- 1) OVERVIEW PAGE ----------
if page == "Overview":
    st.header("Model Summary")
    st.markdown(
        """
**Calibrated Logistic Regression Evaluation (final model)**

**Classification Report:**

- Class 0 (No churn): precision 0.93, recall 0.73, f1-score 0.82  
- Class 1 (Churn): precision 0.53, recall 0.84, f1-score 0.65  

Overall accuracy: 0.76.
        """
    )

    st.subheader("Key Evaluation Plots")
    cols = st.columns(3)
    with cols[0]:
        st.text("PR AUC")
        try:
            st.image(IMG_PR_AUC, caption="PR-AUC")
        except Exception:
            st.info("Image not found.")
    with cols[1]:
        st.text("ROC AUC")
        try:
            st.image(IMG_ROC_AUC, caption="ROC-AUC")
        except Exception:
            st.info("Image not found.")
    with cols[2]:
        st.text("Confusion Matrix")
        try:
            st.image(IMG_CONF_MATRIX, caption="Confusion matrix")
        except Exception:
            st.info("Image not found.")

    st.subheader("SHAP Summary Plot")
    try:
        st.image(IMG_SHAP_SUMMARY, caption="SHAP summary")
    except Exception:
        st.info("Image not found.")

    st.markdown("---")
    st.markdown(
        """
**Design Decisions**

- A logistic regression model was trained on engineered features.
- SMOTE and calibrated probabilities were used for reliable scoring.
- The default threshold is 0.28.
"""
    )

# ---------- 2) PREDICTION & EXPLANATION ----------
elif page == "Prediction & Explanation":
    st.header("Single Customer Prediction & Explanation")
    st.markdown("Enter customer features to get a churn prediction and an explanation.")

    # Input widgets
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Categorical Features")
        paymentmethod = st.selectbox("Payment method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        internetservice = st.selectbox("Internet service", options=["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
        seniorcitizen = st.selectbox("Senior Citizen", options=["Yes", "No"])
        partner = st.selectbox("Partner", options=["Yes", "No"])
        dependents = st.selectbox("Dependents", options=["Yes", "No"])
        paperlessbilling = st.selectbox("Paperless billing", options=["Yes", "No"])
        onlinebackup = st.selectbox("Online backup service", options=["Yes", "No"])
        onlinesecurity = st.selectbox("Online security service", options=["Yes", "No"])
        techsupport = st.selectbox("Tech support service", options=["Yes", "No"])
        deviceprotection = st.selectbox("Device protection service", options=["Yes", "No"])
        streamingtv = st.selectbox("Streaming TV service", options=["Yes", "No"])
        streamingmovies = st.selectbox("Streaming movies service", options=["Yes", "No"])
    with col2:
        st.subheader("Numerical Features")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)
        monthlycharges = st.number_input("Monthly charges", min_value=0.0, max_value=10000.0, value=70.0)
    
    # Derived numerical features
    NumSecurityServices = (1 if onlinesecurity == 'Yes' else 0) + (1 if techsupport == 'Yes' else 0) + (1 if deviceprotection == 'Yes' else 0) + (1 if onlinebackup == 'Yes' else 0)
    NumStreamingServices = (1 if streamingtv == 'Yes' else 0) + (1 if streamingmovies == 'Yes' else 0)
    
    # Derived binary features
    seniorcitizen = 1 if seniorcitizen == 'Yes' else 0
    partner = 1 if partner == 'Yes' else 0
    dependents = 1 if dependents == 'Yes' else 0
    paperlessbilling = 1 if paperlessbilling == 'Yes' else 0
    has_internet = 0 if internetservice == "No" else 1
    is_new_customer = 1 if tenure <= 12 else 0
    is_m2m = 1 if contract == "Month-to-month" else 0
    has_streaming = 1 if NumStreamingServices > 0 else 0
    has_security = 1 if NumSecurityServices > 0 else 0
    is_fiber = 1 if internetservice == "Fiber optic" else 0
    
    input_dict = {
        'paymentmethod': paymentmethod,
        'internetservice': internetservice,
        'contract': contract,
        'seniorcitizen': seniorcitizen,
        'partner': partner,
        'dependents': dependents,
        'paperlessbilling': paperlessbilling,
        'has_internet': has_internet,
        'is_new_customer': is_new_customer,
        'is_m2m': is_m2m,
        'has_streaming': has_streaming,
        'has_security': has_security,
        'is_fiber': is_fiber,
        'tenure': tenure,
        'monthlycharges': monthlycharges,
        'NumSecurityServices': NumSecurityServices,
        'NumStreamingServices': NumStreamingServices,
    }
    
    df_row = pd.DataFrame([input_dict])
    df_row = df_row[FEATURES_ORDERED]

    st.markdown("---")
    threshold_single = st.slider("Decision threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)

    if st.button("Get Prediction & Explain"):
        try:
            prob = safe_predict_proba(model, df_row)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        predicted_label = "will churn" if prob >= threshold_single else "will not churn"
        st.subheader("Prediction Result")
        st.write(f"The customer **{predicted_label}** with a confidence of **{prob:.2%}**.")

        with st.expander("Explanation"):
            st.markdown("SHAP shows how each feature value contributes to the final prediction, helping to understand the model's reasoning.")
            
            try:
                # Separate preprocessor and classifier
                preprocessor = model.estimator.named_steps['preprocessor']
                clf = model.estimator.named_steps['clf']

                # Preprocess the user input
                X_processed = preprocessor.transform(df_row) 

                # Use LinearExplainer
                explainer = shap.LinearExplainer(clf, preprocessor.transform(X_train.sample(50, random_state=42)))

                # Get raw shap_values
                shap_values_array = explainer.shap_values(X_processed)

                # Create an Explanation object
                shap_values = shap.Explanation(
                    values=shap_values_array,
                    base_values=explainer.expected_value,
                    data=X_processed,
                    feature_names=preprocessor.get_feature_names_out()
                )

                # Waterfall plot
                st.write("### SHAP Waterfall Plot")
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")
            
# ---------- 3) BUSINESS IMPACT ----------
elif page == "Business Impact":
    st.header("Business Cost-Sensitive Evaluation (summary)")

    st.markdown("The business costs were defined based on the outcomes of the model's predictions:")
    st.markdown(
        """
- **True Positive:** Saved **$200**
- **False Positive:** Lost **$20**
- **False Negative:** Lost **$500**
- **True Negative:** No gain/loss
        """
    )

    st.subheader("Model results (decision at threshold 0.28)")
    st.markdown(
        """
- **Net gain at 0.28 threshold:** **+$27,040**
- **Net gain per customer:** **+$19.19**
        """
    )
    
    st.subheader("Threshold to maximize net gain")
    st.markdown(
        """
- **Optimal threshold:** **0.03**
- **Maximum net gain:** **$57,760**
        """
    )

# ---------- 4) DOWNLOAD & ABOUT ----------
elif page == "Download & About":
    st.header("About this Model")
    st.markdown("This application demonstrates the final results and prediction capabilities of the churn model.")
    try:
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download saved model (joblib)", data=f, file_name=MODEL_PATH, mime="application/octet-stream")
    except Exception:
        st.info(f"Model file {MODEL_PATH} not found.")

    st.markdown("---")
    st.markdown("**Final Model Metrics**")
    st.markdown(
        """
- Class 0 (No churn): precision 0.93, recall 0.73, f1-score 0.82
- Class 1 (Churn): precision 0.53, recall 0.84, f1-score 0.65
- Overall accuracy: 0.76
        """
    )