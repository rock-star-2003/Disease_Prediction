
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

diseases = pd.read_csv('diseases.csv')
symptoms = pd.read_csv('symptoms.csv')

data = st.dropdown(diseases)