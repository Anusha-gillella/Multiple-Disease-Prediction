#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:13:26 2023

@author: Anusha
"""

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu


# loading the saved models

diabetes_model = pickle.load(open('//Users/vihaan/Desktop/multiple_disease_prediction/saved_models/Diabetes_model.pkl', 'rb'))
diabetes_model_scaler = pickle.load(open('//Users/vihaan/Desktop/multiple_disease_prediction/saved_models/Diabetes_scaler.pkl', 'rb'))

heart_disease_model = pickle.load(open('//Users/vihaan/Desktop/multiple_disease_prediction/saved_models/heart_disease_model.sav','rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction'],
                          default_index = 0)
    
     
     
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    Pregnancies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
   
    
    
# creating a button for Prediction

    diab_diagnosis = ''
   
   
    if st.button('Diabetes Test Result'):
    	X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    	X = diabetes_model_scaler.transform(X)
    	diab_prediction = diabetes_model.predict(X)
    	diab_diagnosis = "The person is Diabetic and Please see an Endocrinologist." if (diab_prediction[0]) else "The person is not Diabetic"
    
    st.success(diab_diagnosis)
    
        



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        sex = st.text_input('Sex')
        cp = st.text_input('Chest Pain types')
        trestbps = st.text_input('Resting Blood Pressure')
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
        
    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        restecg = st.text_input('Resting Electrocardiographic results')
        thalach = st.text_input('Maximum Heart Rate achieved')
        exang = st.text_input('Exercise Induced Angina')
        
    with col3:
        oldpeak = st.text_input('ST depression induced by exercise')
        slope = st.text_input('Slope of the peak exercise ST segment')
        ca = st.text_input('Major vessels colored by flourosopy')
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        Y = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]],dtype=np.float64)
        heart_prediction = heart_disease_model.predict(Y)                          
        if (heart_prediction[0] == 1):
        	heart_diagnosis = 'The person is having heart disease and please see a Cardiologist'
        else:
        	heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
        
    

    
    
    
    
    
