"""
app.py
Streamlit app for Customer Churn Prediction.
Run with: streamlit run app.py  (from inside the src/ folder)
"""
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from preprocessing import preprocess_raw_data

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load model, scaler, and SHAP explainer once ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("../outputs/models/xgboost_tuned.joblib")
    scaler = joblib.load("../outputs/models/scaler.joblib")
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServices"]

st.title("Customer Churn Predictor")
st.write("Predict churn risk for telecom customers, with SHAP-based explanations.")

tab1, tab2 = st.tabs(["🧍 Single Customer", "📁 Batch Upload"])

with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    if st.button("Predict Churn Risk", type="primary"):
        raw_input = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": str(total_charges)
        }])

        processed = preprocess_raw_data(raw_input)
        processed[NUMERIC_COLS] = scaler.transform(processed[NUMERIC_COLS])

        proba = model.predict_proba(processed)[0][1]
        prediction = "🔴 Likely to Churn" if proba >= 0.5 else "🟢 Likely to Stay"

        st.metric("Churn Probability", f"{proba:.1%}")
        st.subheader(prediction)

        # SHAP explanation for this customer
        shap_values = explainer.shap_values(processed)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], processed.iloc[0],
                         matplotlib=True, show=False)
        st.pyplot(fig)




with tab2:
    st.subheader("Upload a CSV of Customers")
    st.write("CSV should have the same columns as the original Telco dataset (including `customerID`, excluding `Churn`).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_raw = pd.read_csv(uploaded_file)
        st.write(f"Loaded {batch_raw.shape[0]} customers.")

        # Keep customerID separately for the final output table (not used in prediction)
        customer_ids = batch_raw["customerID"] if "customerID" in batch_raw.columns else None

        processed_batch = preprocess_raw_data(batch_raw)
        processed_batch[NUMERIC_COLS] = scaler.transform(processed_batch[NUMERIC_COLS])

        probas = model.predict_proba(processed_batch)[:, 1]
        predictions = (probas >= 0.5).astype(int)

        results = pd.DataFrame({
            "customerID": customer_ids if customer_ids is not None else range(len(probas)),
            "Churn_Probability": probas.round(4),
            "Predicted_Churn": ["Yes" if p == 1 else "No" for p in predictions]
        })

        st.dataframe(results.sort_values("Churn_Probability", ascending=False), use_container_width=True)

        st.write(f"**Summary:** {predictions.sum()} of {len(predictions)} customers ({predictions.mean():.1%}) flagged as likely to churn.")

        csv_download = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions as CSV", csv_download,
            file_name="churn_predictions.csv", mime="text/csv"
        )

        # Global SHAP summary for this batch
        st.subheader("What's driving churn risk in this batch?")
        shap_values_batch = explainer.shap_values(processed_batch)
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values_batch, processed_batch, show=False)
        st.pyplot(fig2)