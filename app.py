import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="SmartBank AI",
    page_icon="🏦"
)

st.title("🏦 SmartBank AI")
st.write("Bank Term Deposit Prediction System")

models = joblib.load("smartbank_models.pkl")
model = models["decision_tree"]
preprocessor = models["preprocessor"]

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    age = st.number_input("Age", 18, 100, 35)
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    marital = st.selectbox(
        "Marital Status", ["married", "single", "divorced"]
    )
    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )
    default = st.selectbox("Credit in Default", ["no", "yes"])
    balance = st.number_input("Average Yearly Balance", value=1000)
    housing = st.selectbox("Housing Loan", ["no", "yes"])
    loan = st.selectbox("Personal Loan", ["no", "yes"])
    contact = st.selectbox(
        "Contact Type", ["cellular", "telephone", "unknown"]
    )
    day = st.number_input("Last Contact Day", 1, 31, 15)
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    campaign = st.number_input(
        "Contacts During This Campaign", 1, value=1
    )
    pdays = st.number_input(
        "Days Since Previous Contact (-1 if never contacted)",
        value=-1
    )
    previous = st.number_input(
        "Previous Campaign Contacts", min_value=0, value=0
    )
    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

    submitted = st.form_submit_button("Predict Subscription")

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    prediction = model.predict(
        preprocessor.transform(customer)
    )[0]

    if prediction == 1:
        st.success("Prediction: Customer may subscribe (YES)")
    else:
        st.warning("Prediction: Customer may not subscribe (NO)")