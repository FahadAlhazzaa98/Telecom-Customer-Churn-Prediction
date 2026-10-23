import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    with open("model_config.json") as f:
        config = json.load(f)
    return model, config

model, config = load_model()
threshold = float(config["threshold"])

st.title("📊 Telecom Customer Churn Prediction")
st.write(
    "Predict customer churn risk using "
    "a trained Logistic Regression model."
)

st.info(
    "Educational demonstration using public telecom data. "
    "Predictions are estimates, not guarantees."
)

st.subheader("Customer Information")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", 0.0, 200.0, 70.0
        )
        total = st.number_input(
            "Total Charges", 0.0, 100000.0, 840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = float(
        model.predict_proba(customer)[0, 1]
    )

    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )
    c2.metric(
        "Decision Threshold",
        f"{threshold:.0%}"
    )
    c3.metric(
        "Prediction",
        "Churn" if prediction else "No Churn"
    )

    st.progress(probability)

    if prediction:
        st.warning(
            "This customer is classified as at risk "
            "of churn under the selected threshold."
        )
    else:
        st.success(
            "This customer is classified as lower risk "
            "under the selected threshold."
        )

    st.caption(
        "Risk classification is based on a threshold "
        "selected using validation data."
    )
