import streamlit as st
import pandas as pd
import pickle
import os
import sys

# import from parent directory (project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom preprocessing function
from src.data_cleaning import load_and_clean_data

# Load the trained XGBoost model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
model_path = os.path.abspath(model_path)
model = pickle.load(open(model_path, 'rb'))

# Streamlit UI
st.title("Credit Risk Prediction App")
st.subheader("Enter applicant details:")

# Input fields (
RevolvingUtilizationOfUnsecuredLines = st.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, max_value=2.0, value=0.5)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
NumberOfTime30_59DaysPastDueNotWorse = st.number_input("No. of Times 30–59 Days Past Due", min_value=0, value=0)
DebtRatio = st.number_input("Debt Ratio", min_value=0.0, value=0.3)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
NumberOfOpenCreditLinesAndLoans = st.number_input("No. of Open Credit Lines & Loans", min_value=0, value=3)
NumberOfTimes90DaysLate = st.number_input("No. of Times 90 Days Late", min_value=0, value=0)
NumberRealEstateLoansOrLines = st.number_input("No. of Real Estate Loans or Lines", min_value=0, value=1)
NumberOfTime60_89DaysPastDueNotWorse = st.number_input("No. of Times 60–89 Days Past Due", min_value=0, value=0)
NumberOfDependents = st.number_input("Number of Dependents", min_value=0, value=0)

# Run Prediction
if st.button("Predict Credit Risk"):
    input_df = pd.DataFrame([[
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])

    # You can call a preprocessing function here if needed
    # input_df = preprocess(input_df)

    prediction = model.predict(input_df)[0]
    result = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"
    st.success(f"Prediction: {result}")
