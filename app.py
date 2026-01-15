import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load trained ANN model
model = tf.keras.models.load_model("model.h5")

# Load preprocessor (ColumnTransformer pipeline)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=60000.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Prepare input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Apply SAME preprocessing as training
input_processed = preprocessor.transform(input_data)

# Prediction
prediction = model.predict(input_processed)
prediction_proba = prediction[0][0]

st.subheader(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("The customer is likely to churn")
else:
    st.success("The customer is not likely to churn")
