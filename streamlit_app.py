import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("credit_risk_xgb.pkl")
scaler = joblib.load("credit_risk_scaler.pkl")

# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Risk Prediction App")
st.markdown("Predict whether a loan applicant is High Risk or Low Risk.")
st.divider()

# Layout — two columns
col1, col2 = st.columns([1, 1])

# ---- LEFT COLUMN: Input Form ----
with col1:
    st.subheader("Applicant Details")

    age = st.slider("Age", min_value=21, max_value=65, value=35)
    income = st.number_input("Annual Income (₹)", min_value=20000, max_value=150000, value=60000, step=1000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=5000, max_value=50000, value=20000, step=500)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
    years_employed = st.slider("Years Employed", min_value=0, max_value=20, value=5)
    missed_payments = st.slider("Missed Payments", min_value=0, max_value=10, value=1)

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

# ---- RIGHT COLUMN: Result ----
with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        input_data = np.array([[age, income, loan_amount, credit_score, years_employed, missed_payments]])
        input_scaled = scaler.transform(input_data)

        probability = float(model.predict_proba(input_scaled)[0][1])
        prediction = int(model.predict(input_scaled)[0])
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        if prediction == 1:
            st.error(f"⚠️ {risk_label}")
        else:
            st.success(f"✅ {risk_label}")

        st.metric("Default Probability", f"{round(probability * 100, 2)}%")

        # Store in session state for dashboard
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Age": age,
            "Income": income,
            "Loan Amount": loan_amount,
            "Credit Score": credit_score,
            "Years Employed": years_employed,
            "Missed Payments": missed_payments,
            "Risk": risk_label,
            "Probability": round(probability, 4)
        })

        st.info("Scroll down to see your prediction history dashboard.")

    else:
        st.info("Fill in the applicant details on the left and click Predict Risk.")

st.divider()

# ---- DASHBOARD SECTION ----
st.subheader("📊 Prediction History Dashboard")

if "history" in st.session_state and len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Predictions", len(df))
    m2.metric("High Risk Count", len(df[df["Risk"] == "High Risk"]))
    m3.metric("Low Risk Count", len(df[df["Risk"] == "Low Risk"]))

    st.divider()

    # Chart
    risk_counts = df["Risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    st.bar_chart(risk_counts.set_index("Risk"))

    st.divider()

    # Full history table
    st.dataframe(df, use_container_width=True)

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions yet. Make a prediction above to see the dashboard.")
