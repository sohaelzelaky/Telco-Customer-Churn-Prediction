
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# =========================
# 1. Streamlit page
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# =========================
# 2. Load trained model bundle
# =========================
bundle = joblib.load("churn_model.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
scale_cols = bundle["scale_cols"]
all_features = bundle["all_features"]
selected_features = bundle["selected_features"]

# =========================
# 3. Feature engineering settings
# =========================
ADDONS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies"
]

BINARY_COLS = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "paperless_billing"
]

BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "Female": 1,
    "Male": 0
}

NOMINAL = {
    "internet_service": ["DSL", "Fiber optic", "No"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "payment_method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
}

def engineer_features(d):
    d = d.copy()

    # Normalize service values to match training encoding
    d = d.replace({
        "No internet service": "No",
        "No phone service": "No"
    })

    # Feature creation
    d["num_addons"] = (d[ADDONS] == "Yes").sum(axis=1)

    d["avg_monthly_charge"] = (
        d["total_charges"] / d["tenure"].replace(0, 1)
    )

    d["charge_change"] = (
        d["monthly_charges"] - d["avg_monthly_charge"]
    )

    d["has_family"] = (
        (d["partner"] == "Yes") |
        (d["dependents"] == "Yes")
    ).astype(int)

    # Binary encoding
    for c in BINARY_COLS:
        d[c] = d[c].map(BINARY_MAP).astype(int)

    # One-hot encoding
    for c, cats in NOMINAL.items():
        dummies = pd.get_dummies(
            pd.Categorical(d[c], categories=cats),
            prefix=c,
            dtype=int
        )

        d = pd.concat(
            [d.drop(columns=c), dummies],
            axis=1
        )

    # Clean feature names
    d.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_")
        for c in d.columns
    ]

    return d


# =========================
# 4. App title
# =========================
st.title("Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to leave the company."
)

st.divider()

st.subheader("Customer Information")

# =========================
# 5. Customer input form
# =========================
with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])

        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        partner = st.selectbox("Partner", ["Yes", "No"])

        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            NOMINAL["payment_method"]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# =========================
# 6. Prediction
# =========================
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }])

    # Feature engineering
    X = engineer_features(customer)

    # Match training feature order
    X = X.reindex(columns=all_features, fill_value=0)

    # Apply saved scaler
    X[scale_cols] = scaler.transform(X[scale_cols])

    # Select trained features
    X = X[selected_features]

    # Prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    churn_probability = probability[1] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "OUT — This customer is likely to leave the company."
        )
        st.markdown("### Churn Status: OUT")

    else:
        st.success(
            "IN — This customer is likely to stay with the company."
        )
        st.markdown("### Churn Status: IN")

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability:.2f}%"
    )

    st.progress(float(churn_probability / 100))

    st.caption(
        "IN = Customer stays with the company | "
        "OUT = Customer leaves the company"
    )
