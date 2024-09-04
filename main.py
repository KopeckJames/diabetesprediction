
import streamlit as st
from streamlit import streamlit.cli
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from joblib import dump, load
st.title('Diabetes Prediction Model')

st.info('This is an app that will detect if you possibly have diabetes based on a number of common medical data and information!')
st.write('**Raw data**')
df = pd.read_csv('best_features.csv')
df
model = load('diabetes.joblib')
ethnicity_map = {
    "Caucasian": 0,
    "African American": 1,
    "Asian": 2,
    "Other": 3
}
gender_map = {
  "Male": 0,
  "Female":1
}
yesno_map= {
  'Yes':1,
  'No':0
}
status_map={
  'Low': 0,
  'Middle':1,
  'High':2

}


# Input features
with st.sidebar:
  st.header('Input features')
  Age = st.number_input('Age')
  Gender = st.selectbox('Gender',('Male','Female'))
  Gender_value = gender_map[Gender]
  Ethnicity= st.selectbox('Ethnicity', ('Caucasian', 'African American', 'Asian','Other'))
  ethnicity_value = ethnicity_map[Ethnicity]
  SocioeconomicStatus = st.selectbox('Economic Status',('Low','Middle','High'))
  SocioeconomicStatus_value = status_map[SocioeconomicStatus]
  BMI = st.number_input('Body Mass Index - Range 15-40')
  Smoking = st.selectbox('Smoker?',('Yes','No'))
  Smoking_value = yesno_map[Smoking]
  DietQuality = st.number_input('Diet Quality- Range 0-10')
  FamilyHistoryDiabetes = st.selectbox('Is there a family history of Diabetes?',('Yes','No'))
  FamilyHistoryDiabetes_value = yesno_map[FamilyHistoryDiabetes]
  GestationalDiabetes = st.selectbox('Is there a history of Gestational Diabetes?',('Yes','No'))
  GestationalDiabetes_value = yesno_map[FamilyHistoryDiabetes]
  PolycysticOvarySyndrome = st.selectbox('Presence of polycystic ovary syndrome?',('Yes','No'))
  PolycysticOvarySyndrome_value = yesno_map[PolycysticOvarySyndrome] 
  PreviousPreDiabetes = st.selectbox(' History of previous pre-diabetes?',('Yes','No'))
  PreviousPreDiabetes_value = yesno_map[PreviousPreDiabetes] 
  Hypertension = st.selectbox('Presence of hypertension?',('Yes','No'))
  Hypertension_value = yesno_map[Hypertension] 
  SystolicBP = st.number_input('Systolic blood pressure, ranging from 90 to 180 mmHg')
  DiastolicBP = st.number_input('Diastolic blood pressure, ranging from 60 to 120 mmHg')
  FastingBloodSugar = st.number_input('Fasting blood sugar levels, ranging from 70 to 200 mg/dL.')
  HbA1c = st.number_input('Hemoglobin A1c levels, ranging from 4.0% to 10.0%')
  SerumCreatinine = st.number_input('Serum creatinine levels, ranging from 0.5 to 5.0 mg/dL')
  BUNLevels = st.number_input('Blood Urea Nitrogen levels, ranging from 5 to 50 mg/dL')
  CholesterolTotal = st.number_input('Total cholesterol levels, ranging from 150 to 300 mg/dL.')
  CholesterolLDL = st.number_input('Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL')
  CholesterolHDL = st.number_input('High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL')
  CholesterolTriglycerides =st.number_input('Triglycerides levels, ranging from 50 to 400 mg/dL')
  AntihypertensiveMedications = st.selectbox('Use of antihypertensive medications',('Yes','No'))
  Statins = st.selectbox('Use of statins',('Yes','No'))
  Statins_value = yesno_map[Statins] 
  AntidiabeticMedications = st.selectbox('Use of antidiabetic medications',('Yes','No'))
  
  AntidiabeticMedications_value = yesno_map[AntidiabeticMedications] 
  FrequentUrination = st.selectbox('Presence of frequent urination',('Yes','No')) 
  FrequentUrination_value = yesno_map[FrequentUrination] 
  ExcessiveThirst = st.selectbox('Presence of excessive thirst',('Yes','No'))
  UnexplainedWeightLoss = st.selectbox(' Presence of unexplained weight loss',('Yes','No'))
  UnexplainedWeightLoss_value = yesno_map[UnexplainedWeightLoss] 
  FatigueLevels = st.number_input('Fatigue levels, ranging from 0 to 10')
  BlurredVision = st.selectbox('Presence of blurred vision',('Yes','No'))
  TinglingHandsFeet = st.selectbox('Presence of tingling in hands or feet?',('Yes','No'))
  TinglingHandsFeet_value = yesno_map[TinglingHandsFeet] 
  QualityOfLifeScore = st.number_input('Quality of life score, ranging from 0 to 100')
  MedicationAdherence = st.number_input(' Medication adherence score, ranging from 0 to 10')
  HealthLiteracy = st.number_input('Health literacy score, ranging from 0 to 10')


  # Create a DataFrame for the input features
  data = {'Age':Age,
          'Gender': Gender,
          'Ethnicity': Ethnicity, 
          'SocioeconomicStatus': SocioeconomicStatus, 
          'BMI': BMI, 
          'Smoking': Smoking, 
          'DietQuality': DietQuality, 
          'FamilyHistoryDiabetes': FamilyHistoryDiabetes,
          'GestationalDiabetes': GestationalDiabetes, 
          'PolycysticOvarySyndrome': PolycysticOvarySyndrome, 
          'PreviousPreDiabetes': PreviousPreDiabetes,
          'Hypertension' : Hypertension, 
          'SystolicBP': SystolicBP, 
          'DiastolicBP' : DiastolicBP, 
          'FastingBloodSugar': FastingBloodSugar,
          'HbA1c': HbA1c, 
          'SerumCreatinine': SerumCreatinine, 
          'BUNLevels': BUNLevels, 
          'CholesterolTotal': CholesterolTotal,
          'CholesterolLDL' : CholesterolLDL, 
          'CholesterolHDL' : CholesterolHDL, 
          'CholesterolTriglycerides': CholesterolTriglycerides,
          'AntihypertensiveMedications' : AntihypertensiveMedications, 
          'Statins' : Statins, 
          'AntidiabeticMedications': AntidiabeticMedications,
          'FrequentUrination' : FrequentUrination, 
          'ExcessiveThirst' : ExcessiveThirst, 
          'UnexplainedWeightLoss' : UnexplainedWeightLoss,
          'FatigueLevels': FatigueLevels, 
          'BlurredVision': BlurredVision, 
          'TinglingHandsFeet' : TinglingHandsFeet, 
          'QualityOfLifeScore': QualityOfLifeScore,  
          'MedicationAdherence' : MedicationAdherence, 
          'HealthLiteracy' : HealthLiteracy,
          }
input_df = pd.DataFrame([data])

# Make predictions using the model
streamlit run main.py