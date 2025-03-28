# import streamlit as st
import joblib
import numpy as np
import statistics
import pickle
import streamlit as st
import pandas as pd

# Load the model using pickle

knn_model = joblib.load("knn_model_predict_desease.pkl")
nb_model = joblib.load("nb_model_predict_desease.pkl")
svm_model = joblib.load("svm_model_predict_desease.pkl")


# df = pd.read_csv('.\\Dataset\\Training.csv').iloc[:,:-1]
# print(df)

# symptoms = df.iloc[:,:-1].columns

# deseases = df['prognosis'].unique()

# print(deseases)

# # Save symptoms to a CSV file
# symptoms_df = pd.DataFrame(symptoms, columns=['Symptom'])
# symptoms_df.to_csv('symptoms.csv', index=False)

# # Save diseases to a CSV file
# diseases_df = pd.DataFrame(deseases, columns=['Disease'])
# diseases_df.to_csv('diseases.csv', index=False)

diseases = pd.read_csv('diseases.csv')
symptoms = pd.read_csv('symptoms.csv')

st.text('hello everyone')