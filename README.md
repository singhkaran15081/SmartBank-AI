# SmartBank AI

A Machine Learning project that predicts whether a bank customer is likely to subscribe to a term deposit.

## Features
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature encoding
- Logistic Regression
- Balanced Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Model comparison using classification metrics
- Interactive Streamlit web application

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib / Seaborn (for analysis and visualization)

## Dataset
Bank Marketing dataset from the UCI Machine Learning Repository.

Place the dataset file `bank-full.csv` in the project folder before training.

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Train the models and generate the saved model file:

   python check_data.py

3. Start the Streamlit application:

   streamlit run app.py

## Project Goal
To analyze customer data and predict potential term-deposit subscriptions using Machine Learning.