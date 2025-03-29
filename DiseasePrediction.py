
import joblib
import numpy as np
import statistics
import pickle
import streamlit as st
import pandas as pd
import statistics
from sklearn.preprocessing import LabelEncoder
encorder = LabelEncoder()

# Load the model using pickle

knn_model = joblib.load("knn_model_predict_desease.pkl")
nb_model = joblib.load("nb_model_predict_desease.pkl")
svm_model = joblib.load("svm_model_predict_desease.pkl")
encoder = joblib.load("encoder.pkl")


    
diseases = pd.read_csv('diseases.csv')
symptoms = [j.replace('_',' ') for i in pd.read_csv('symptoms.csv').values.tolist() for j in i]
inp_symptoms = st.multiselect("symptom_index" , symptoms)
symptoms_index = {}
for index,val in enumerate(symptoms):
    symptoms = val
    symptoms_index[val] = index
    
data_dict = {
    "symptom_index" : symptoms_index,
    'classes' : encoder.classes_
}
    
def predict_disease (symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    
    for symtom in symptoms:
        index = data_dict["symptom_index"][symtom]
        input_data[index] = 1
    
    input_data = np.array(input_data).reshape(1,-1)
    
    Knn_predict = data_dict['classes'][knn_model.predict(input_data)][0]
    nb_predict = data_dict['classes'][ nb_model.predict(input_data)][0]
    svm_predict = data_dict['classes'][svm_model.predict(input_data)][0]

    final_pred = statistics.mode([Knn_predict,nb_predict,svm_predict])
    return final_pred



if st.button('find Disease'):
    st.text(predict_disease(inp_symptoms))