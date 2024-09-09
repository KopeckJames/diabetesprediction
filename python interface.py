
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import numpy as np
import random
import gradio as gr


best_feat = pd.read_csv('best_features.csv')
best_feat

choices_gender = [0, 1]
choices_ethn= [0,1,2,3]
choices_socio= [0,1,2]
choices_water=[0,1]
choices_yesno=[0,1]


X = best_feat.drop('Diagnosis', axis=1)
y = best_feat['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = CatBoostClassifier(n_estimators=1500, learning_rate=.02)

# Fit the model to the training data
model.fit(X_train, y_train)



def diabetes(Age, Gender, Ethnicity, SocioeconomicStatus,
       BMI, Smoking,
       DietQuality, FamilyHistoryDiabetes,
       GestationalDiabetes, PolycysticOvarySyndrome, PreviousPreDiabetes,
       Hypertension, SystolicBP, DiastolicBP, FastingBloodSugar,
       HbA1c, SerumCreatinine, BUNLevels, CholesterolTotal,
       CholesterolLDL, CholesterolHDL, CholesterolTriglycerides,
       AntihypertensiveMedications, Statins, AntidiabeticMedications,
       FrequentUrination, ExcessiveThirst, UnexplainedWeightLoss,
       FatigueLevels, BlurredVision,
       TinglingHandsFeet, QualityOfLifeScore, HeavyMetalsExposure, WaterQuality,
       MedicationAdherence, HealthLiteracy):
#turning the arguments into a numpy array  

       x = np.array([Age, Gender, Ethnicity, SocioeconomicStatus,
       BMI, Smoking,
       DietQuality, FamilyHistoryDiabetes,
       GestationalDiabetes, PolycysticOvarySyndrome, PreviousPreDiabetes,
       Hypertension, SystolicBP, DiastolicBP, FastingBloodSugar,
       HbA1c, SerumCreatinine, BUNLevels, CholesterolTotal,
       CholesterolLDL, CholesterolHDL, CholesterolTriglycerides,
       AntihypertensiveMedications, Statins, AntidiabeticMedications,
       FrequentUrination, ExcessiveThirst, UnexplainedWeightLoss,
       FatigueLevels, BlurredVision,
       TinglingHandsFeet, QualityOfLifeScore, HeavyMetalsExposure, WaterQuality,
       MedicationAdherence, HealthLiteracy])
       
       x=scaler.transform(x.reshape(1,-1))
       prediction = model.predict(x)

       return "Healthy" if prediction[0]==0 else "Get checked for Diabetes"


#Insturctor model
patient_x={
'Age'  : random.randint(18, 100),  # Assuming age range is 18-100
'Gender'   : random.choice(choices_gender),
'Ethnicity' :random.choice(choices_ethn),
'SocioeconomicStatus' :random.choice(choices_socio),
'BMI' :random.randint(15, 40),
'Smoking' :random.choice(choices_yesno),
'DietQuality' :random.randint(0, 10),
'FamilyHistoryDiabetes' :random.choice(choices_yesno),
'GestationalDiabetes' :random.choice(choices_yesno),
'PolycysticOvarySyndrome' :random.choice(choices_yesno),
'PreviousPreDiabetes' :random.choice(choices_yesno),
'Hypertension' :random.choice(choices_yesno),
'SystolicBP' :random.randint(90, 180),
'DiastolicBP' :random.randint(60, 120),
'FastingBloodSugar' :random.randint(70, 200),
'HbA1c' : round(random.uniform(4.0, 10.0), 2),
'SerumCreatinine' : round(random.uniform(0.5, 5.0), 2),
'BUNLevels' :random.randint(5, 50),
'CholesterolTotal' :random.randint(150, 300),
'CholesterolLDL' :random.randint(50, 200),
'CholesterolHDL' :random.randint(20, 100),
'CholesterolTriglycerides' :random.randint(50, 400),
'AntihypertensiveMedications' :random.choice(choices_yesno),
'Statins' :random.choice(choices_yesno),
'AntidiabeticMedications' :random.choice(choices_yesno),
'FrequentUrination' :random.choice(choices_yesno),
'ExcessiveThirst' :random.choice(choices_yesno),
'UnexplainedWeightLoss' :random.choice(choices_yesno),
'FatigueLevels' :random.randint(0, 10),
'BlurredVision' :random.choice(choices_yesno),
'TinglingHandsFeet' :random.choice(choices_yesno),
'QualityOfLifeScore' :random.randint(0, 100),
'HeavyMetalsExposure' :random.choice(choices_yesno),
'WaterQuality' :random.choice(choices_water),
'MedicationAdherence' :random.randint(0, 10),
'HealthLiteracy' :random.randint(0, 10)}

def main():
    with gr.Blocks() as demo:
    # Create buttons and functions
        button1 = gr.Button("Group 1's Patient")
        button2 = gr.Button("Group 3's Patient")
        button3 = gr.Button("Instructor's Patient")

    # Create output space
        model_output = gr.Textbox(label="Diagnosis")
       

        def run_model1(button):
 
            result = diabetes(69,0,0,0,31.74,0,1.48,0,1,0,0,1,108,66,72.42,6.9,3.01,24.01,218.11,81.12,24.16,263.81,0,0,1,1,0,1,9.14,0,1,73.23,0,0,3.71,5.65)
            

            # Display input data and model output
            
            model_output = str(result)

            return result

        def run_model2(button):
            
            result2 = diabetes(87,0,0,0,30.82,0,0.61,1,0,0,0,0,137,66,80.03,7.29,4.89,48.64,160.35,105.09,54.06,343.82,0,1,0,0,0,0,1.53,0,0,58.4,0,0,7.43,9.44)

            # Display input data and model output
            
            model_output = str(result2)
            return result2

        def run_model3(button):
            
            result = diabetes(**patient_x)

            # Display input data and model output
           
            model_output = str(result)
            return result
        # Connect buttons to functions
        button1.click(run_model1, outputs=[model_output])
        button2.click(run_model2, outputs=[model_output])
        button3.click(run_model3, outputs=[model_output])
        demo.launch(share=True)



