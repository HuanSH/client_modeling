import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# Load the pre-trained model
#load the trained model 
model = load_model('/home/oscar/resources/simpleANN_keras/simple_ann_model.h5') 

#load the encoders and scaler
with open('/home/oscar/resources/simpleANN_keras/label_encoder_gender.pkl', 'rb') as f:# rb means read binary
    label_encoder_gender = pickle.load(f)

with open('/home/oscar/resources/simpleANN_keras/onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

with open('/home/oscar/resources/simpleANN_keras/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
data_section, prediction_section = st.columns(2, gap="large")
with data_section:
    st.title("Customer Churn Predictor")
    st.write("This app predicts whether a customer will churn based on their features.")
    
    # Input features
    credit_score = st.slider("Credit Score", 350, 850, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Prepare input data for prediction
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

# Encode categorical features
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
geography_encoded = onehot_encoder_geography.transform(input_data[['Geography']]).toarray()
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1), geography_df], axis=1)
# Scale features
input_data_scaled = scaler.transform(input_data)
# Make prediction
prediction = model.predict(input_data_scaled)

with prediction_section:
   st.header("Prediction Result")
   st.write(f"Churn Probability: {prediction[0][0]:.2f}")
