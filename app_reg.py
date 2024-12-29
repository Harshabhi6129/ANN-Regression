import streamlit as st 
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

model = tf.keras.models.load_model('Regression_model.keras')

with open('OneHotEncoder_geo.pkl','rb') as file:
    OneHotEncoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
st.title('Estimated salary prediction')

geography = st.selectbox('Geography', OneHotEncoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_number = st.selectbox('Is Active Number', [0,1])

input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure' : [tenure],
    'Balance':[balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_number],
    'Exited': [exited]    
}

geo_encoded = OneHotEncoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHotEncoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encoded columns with encoded data
input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

st.write(f'Predicted salary: {prediction_salary :.2f}')