import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler


st.title("DIABETES PREDICTION")
from PIL import Image
image = Image.open('dia4.jpg')
st.image(image,width = 400)
st.write("")
st.write("")



Pregnancies= st.text_input("Number of Pregnancies [eg.0]","")
Glucose = st.text_input(" Enter Glucose Level in (mg/dl) [eg. 80]","")
BloodPressure = st.text_input("Enter the Blood Pressure in (mm/Hg) [eg. 90]","")
SkinThickness = st.text_input("Enter the Skin Thickness in mm [eg. 20]","")
Insulin = st.text_input("Enter Insulin level in (IU/ml) [eg. 80]","")
BMI = st.text_input("Enter BMI in (kg/m2) [eg. 23.1]","")
DPF = st.text_input("Enter DiabetesPedigreeFunction [eg. 0.52]","")
Age = st.text_input("Enter Age [eg. 32]","")

pickle_in = open("diabetes.pkl","rb")
tree = pickle.load(pickle_in)


Pregnancies = int(Pregnancies)
Glucose  = float(Glucose)
BloodPressure = float(BloodPressure)
SkinThickness = float(SkinThickness)
Insulin = float(Insulin)
BMI = float(BMI)
DPF = float(DPF)
Age = int(Age)





features = pd.DataFrame({
        "Pregnancies" : Pregnancies,
        "Glucose" : Glucose, 
        "BloodPressure " : BloodPressure ,
        "SkinThickness " : SkinThickness , 
        "Insulin " : Insulin , 
        "BMI " : BMI ,
        "DPF " : DPF , 
        "Age" : Age
    }, index=[0])

#st.write(features)

if st.button("Predict"):
     x = features
     #st.write(x)
     pred = tree.predict(features)[0]
     #st.write(pred)
     
     if pred:
          image1 = Image.open('patient.jpg')
          st.image(image1,width = 300)
          st.success('Oops! You have diabetes.')
  
     else:
          image2 = Image.open('doctor.jpg')
          st.image(image2,width = 300)
          st.success("Great! You don't have diabetes.")
  
    
    
    
    
    
  
    
    

   
    

    
