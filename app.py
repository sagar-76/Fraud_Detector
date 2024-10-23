import streamlit as st
import pandas as pd
import pickle  # Use pickle to load the model

# Load your trained model
with open('fraud_detection_model.pkl', 'rb') as f:  # Update with your model's path
    model = pickle.load(f)

# Define the feature names used during model training in the specified order
feature_names = [
    'step',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'isFlaggedFraud',  # This will be computed based on the amount
    'amount_capped',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
    'hour_of_day',
]

# Function to compute class weights
def compute_class_weight(amount_capped):
    if amount_capped > 400000:
        return 5
    elif amount_capped > 200000:
        return 2
    else:
        return 1

# Streamlit app layout
st.title("Fraud Detection Prediction")

# Collect user input
step = st.number_input("Enter step (hour of time):", min_value=0)
oldbalanceOrg = st.number_input("Enter old balance origin:")
newbalanceOrig = st.number_input("Enter new balance origin:")
oldbalanceDest = st.number_input("Enter old balance destination:")
newbalanceDest = st.number_input("Enter new balance destination:")
amount_capped = st.number_input("Enter amount capped:")
type_CASH_OUT = st.selectbox("Enter type_CASH_OUT (0 or 1):", options=[0, 1])
type_DEBIT = st.selectbox("Enter type_DEBIT (0 or 1):", options=[0, 1])
type_PAYMENT = st.selectbox("Enter type_PAYMENT (0 or 1):", options=[0, 1])
type_TRANSFER = st.selectbox("Enter type_TRANSFER (0 or 1):", options=[0, 1])
hour_of_day = st.number_input("Enter hour of day (0-23):", min_value=0, max_value=23)

# Calculate isFlaggedFraud based on amount_capped
isFlaggedFraud = 1 if amount_capped > 200000 else 0

# Prepare input DataFrame
user_input = {
    'step': step,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'isFlaggedFraud': isFlaggedFraud,
    'amount_capped': amount_capped,
    'type_CASH_OUT': type_CASH_OUT,
    'type_DEBIT': type_DEBIT,
    'type_PAYMENT': type_PAYMENT,
    'type_TRANSFER': type_TRANSFER,
    'hour_of_day': hour_of_day,
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])[feature_names]

# Display the flagged fraud status
st.write(f"Flagged as fraud based on amount: {isFlaggedFraud}")

# Predict the fraud status using the model
if st.button("Predict"):
    predicted_fraud = model.predict(input_df)
    st.write(f"Predicted fraud status: {predicted_fraud[0]}")
