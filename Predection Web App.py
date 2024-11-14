# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:31:11 2024

@author: suhas
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('D:\Mini-Project-CKDP/trained_model.sav', 'rb'))


# creating a function for Prediction

def ckd_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person does not have CKD'
    else:
      return 'The person has been diagnosed with CKD'
  
    
  
def main():
    
    
    # giving a title
    st.title('CKD Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Creatininelevel = st.text_input('Creatinine level')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('CKD Test Result'):
        diagnosis = ckd_prediction([Creatininelevel, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()