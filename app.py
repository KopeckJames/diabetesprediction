import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import numpy as np
import random
import gradio as gr

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from gradio.themes import Soft, Glass






best_feat = pd.read_csv('best_features.csv')
best_feat




X = best_feat.drop('Diagnosis', axis=1)
y = best_feat['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = CatBoostClassifier(n_estimators=1880, learning_rate=.02)

# Fit the model to the training data
model.fit(X_train, y_train)


#### define diabetes function



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
       
       x= scaler.transform(x.reshape(1,-1))
       prediction = model.predict(x)

       return "Healthy" if prediction[0]==0 else "Get checked for Diabetes"

def create_random_patient():
    return {
        'Age': random.randint(18, 100),
        'Gender': random.choice(["Male", "Female"]),
        'Ethnicity': random.choice(["Caucasian", "African American", "Asian", "Hispanic"]),
        'SocioeconomicStatus': random.choice(["Low", "Middle", "High"]),
        'BMI': round(random.uniform(15, 40), 2),
        'Smoking': random.choice([True, False]),
        'DietQuality': random.randint(0, 10),
        'FamilyHistoryDiabetes': random.choice([True, False]),
        'GestationalDiabetes': random.choice([True, False]),
        'PolycysticOvarySyndrome': random.choice([True, False]),
        'PreviousPreDiabetes': random.choice([True, False]),
        'Hypertension': random.choice([True, False]),
        'SystolicBP': random.randint(90, 180),
        'DiastolicBP': random.randint(60, 120),
        'FastingBloodSugar': random.randint(70, 200),
        'HbA1c': round(random.uniform(4.0, 10.0), 2),
        'SerumCreatinine': round(random.uniform(0.5, 5.0), 2),
        'BUNLevels': random.randint(5, 50),
        'CholesterolTotal': random.randint(150, 300),
        'CholesterolLDL': random.randint(50, 200),
        'CholesterolHDL': random.randint(20, 100),
        'CholesterolTriglycerides': random.randint(50, 400),
        'AntihypertensiveMedications': random.choice([True, False]),
        'Statins': random.choice([True, False]),
        'AntidiabeticMedications': random.choice([True, False]),
        'FrequentUrination': random.choice([True, False]),
        'ExcessiveThirst': random.choice([True, False]),
        'UnexplainedWeightLoss': random.choice([True, False]),
        'FatigueLevels': random.randint(0, 10),
        'BlurredVision': random.choice([True, False]),
        'TinglingHandsFeet': random.choice([True, False]),
        'QualityOfLifeScore': random.randint(0, 100),
        'HeavyMetalsExposure': random.choice([True, False]),
        'WaterQuality': random.choice(["Poor", "Good"]),
        'MedicationAdherence': random.randint(0, 10),
        'HealthLiteracy': random.randint(0, 10)
    }
def create_feature_importance_plot(model, feature_names):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    return plt

def create_prediction_probability_plot(probabilities):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=['Healthy', 'Diabetes Risk'], y=probabilities)
    plt.title('Prediction Probability')
    plt.ylim(0, 1)
    plt.tight_layout()
    return plt

def predict_diabetes(*args):
    # Convert inputs to the format expected by your model
    inputs = list(args)
    
    # Convert categorical variables to numerical
    inputs[1] = 0 if inputs[1] == "Male" else 1  # Gender
    inputs[2] = ["Caucasian", "African American", "Asian", "Hispanic"].index(inputs[2])  # Ethnicity
    inputs[3] = ["Low", "Middle", "High"].index(inputs[3])  # Socioeconomic Status
    inputs[33] = 0 if inputs[33] == "Poor" else 1  # Water Quality
    
    # Convert boolean inputs to integers
    bool_indices = [5, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]
    for idx in bool_indices:
        inputs[idx] = int(inputs[idx])
    
    # Ensure all inputs are float
    inputs = [float(x) for x in inputs]
    
    # Make prediction
    x = np.array(inputs)
    x = scaler.transform(x.reshape(1, -1))
    prediction_proba = model.predict_proba(x)[0]
    prediction = "High Risk - Get checked for Diabetes" if prediction_proba[1] > 0.5 else "Low Risk - Likely Healthy"
    
    # Create plots
    feature_importance_fig = create_feature_importance_plot(model, X.columns)
    prediction_probability_fig = create_prediction_probability_plot(prediction_proba)
    
    return prediction, feature_importance_fig, prediction_probability_fig

my_theme = gr.Theme.from_hub("ParityError/Interstellar")
def create_improved_interface():
    with gr.Blocks(theme=my_theme) as demo:
        gr.Markdown("# Comprehensive Diabetes Risk Assessment")
        
        with gr.Tab("Patient Information"):
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 100, label="Age")
                    gender = gr.Radio(["Male", "Female"], label="Gender")
                    ethnicity = gr.Dropdown(["Caucasian", "African American", "Asian", "Hispanic"], label="Ethnicity")
                    socioeconomic_status = gr.Radio(["Low", "Middle", "High"], label="Socioeconomic Status")
                with gr.Column():
                    bmi = gr.Slider(15, 50, label="BMI")
                    smoking = gr.Checkbox(label="Smoking")
                    diet_quality = gr.Slider(0, 10, label="Diet Quality")
                    family_history = gr.Checkbox(label="Family History of Diabetes")
        
        with gr.Tab("Medical History"):
            with gr.Row():
                with gr.Column():
                    gestational_diabetes = gr.Checkbox(label="Gestational Diabetes")
                    pcos = gr.Checkbox(label="Polycystic Ovary Syndrome")
                    prediabetes = gr.Checkbox(label="Previous Pre-Diabetes")
                    hypertension = gr.Checkbox(label="Hypertension")
                with gr.Column():
                    systolic_bp = gr.Slider(90, 200, label="Systolic Blood Pressure")
                    diastolic_bp = gr.Slider(60, 120, label="Diastolic Blood Pressure")
                    fasting_glucose = gr.Slider(70, 200, label="Fasting Blood Sugar")
                    hba1c = gr.Slider(4, 14, label="HbA1c")
        
        with gr.Tab("Lab Results"):
            with gr.Row():
                with gr.Column():
                    serum_creatinine = gr.Slider(0.5, 3.0, label="Serum Creatinine")
                    bun_levels = gr.Slider(5, 50, label="BUN Levels")
                    cholesterol_total = gr.Slider(100, 300, label="Total Cholesterol")
                    cholesterol_ldl = gr.Slider(50, 250, label="LDL Cholesterol")
                with gr.Column():
                    cholesterol_hdl = gr.Slider(20, 100, label="HDL Cholesterol")
                    triglycerides = gr.Slider(50, 500, label="Triglycerides")
        
        with gr.Tab("Medications and Symptoms"):
            with gr.Row():
                with gr.Column():
                    antihypertensive_medications = gr.Checkbox(label="Taking Antihypertensive Medications")
                    statins = gr.Checkbox(label="Taking Statins")
                    antidiabetic_medications = gr.Checkbox(label="Taking Antidiabetic Medications")
                    frequent_urination = gr.Checkbox(label="Frequent Urination")
                    excessive_thirst = gr.Checkbox(label="Excessive Thirst")
                with gr.Column():
                    unexplained_weight_loss = gr.Checkbox(label="Unexplained Weight Loss")
                    fatigue_levels = gr.Slider(0, 10, label="Fatigue Levels")
                    blurred_vision = gr.Checkbox(label="Blurred Vision")
                    tingling_hands_feet = gr.Checkbox(label="Tingling in Hands/Feet")
        
        with gr.Tab("Lifestyle and Environment"):
            with gr.Row():
                with gr.Column():
                    quality_of_life = gr.Slider(0, 100, label="Quality of Life Score")
                    heavy_metals_exposure = gr.Checkbox(label="Heavy Metals Exposure")
                    water_quality = gr.Radio(["Poor", "Good"], label="Water Quality")
                with gr.Column():
                    medication_adherence = gr.Slider(0, 10, label="Medication Adherence")
                    health_literacy = gr.Slider(0, 10, label="Health Literacy")
        
        with gr.Row():
            submit_btn = gr.Button("Predict")
            
            sample_btn = gr.Button("Sample Data")
        
        with gr.Row():
            prediction = gr.Textbox(label="Diabetes Risk Assessment")
        
        with gr.Row():
            with gr.Column():
                feature_importance_plot = gr.Plot(label="Top 10 Feature Importances")
            with gr.Column():
                prediction_probability_plot = gr.Plot(label="Prediction Probability")
        
        gr.Markdown("## Disclaimer")
        gr.Markdown("This tool is for informational purposes only and should not be considered medical advice. Please consult with a healthcare professional for proper diagnosis and treatment.")
        
        def fill_sample_data():
            sample_data = create_random_patient()
            return [
                sample_data['Age'], sample_data['Gender'], sample_data['Ethnicity'], sample_data['SocioeconomicStatus'],
                sample_data['BMI'], sample_data['Smoking'], sample_data['DietQuality'], sample_data['FamilyHistoryDiabetes'],
                sample_data['GestationalDiabetes'], sample_data['PolycysticOvarySyndrome'], sample_data['PreviousPreDiabetes'],
                sample_data['Hypertension'], sample_data['SystolicBP'], sample_data['DiastolicBP'],
                sample_data['FastingBloodSugar'], sample_data['HbA1c'], sample_data['SerumCreatinine'],
                sample_data['BUNLevels'], sample_data['CholesterolTotal'], sample_data['CholesterolLDL'],
                sample_data['CholesterolHDL'], sample_data['CholesterolTriglycerides'],
                sample_data['AntihypertensiveMedications'], sample_data['Statins'], sample_data['AntidiabeticMedications'],
                sample_data['FrequentUrination'], sample_data['ExcessiveThirst'], sample_data['UnexplainedWeightLoss'],
                sample_data['FatigueLevels'], sample_data['BlurredVision'], sample_data['TinglingHandsFeet'],
                sample_data['QualityOfLifeScore'], sample_data['HeavyMetalsExposure'], sample_data['WaterQuality'],
                sample_data['MedicationAdherence'], sample_data['HealthLiteracy']
            ]

        sample_btn.click(
            fill_sample_data,
            outputs=[age, gender, ethnicity, socioeconomic_status, bmi, smoking, diet_quality, family_history,
                     gestational_diabetes, pcos, prediabetes, hypertension, systolic_bp, diastolic_bp,
                     fasting_glucose, hba1c, serum_creatinine, bun_levels, cholesterol_total, cholesterol_ldl,
                     cholesterol_hdl, triglycerides, antihypertensive_medications, statins, antidiabetic_medications,
                     frequent_urination, excessive_thirst, unexplained_weight_loss,
                     fatigue_levels, blurred_vision, tingling_hands_feet, quality_of_life, heavy_metals_exposure,
                     water_quality, medication_adherence, health_literacy]
        )

        submit_btn.click(
            predict_diabetes,
            inputs=[age, gender, ethnicity, socioeconomic_status, bmi, smoking, diet_quality, family_history,
                    gestational_diabetes, pcos, prediabetes, hypertension, systolic_bp, diastolic_bp,
                    fasting_glucose, hba1c, serum_creatinine, bun_levels, cholesterol_total, cholesterol_ldl,
                    cholesterol_hdl, triglycerides, antihypertensive_medications, statins, antidiabetic_medications,
                    frequent_urination, excessive_thirst, unexplained_weight_loss,
                    fatigue_levels, blurred_vision, tingling_hands_feet, quality_of_life, heavy_metals_exposure,
                    water_quality, medication_adherence, health_literacy],
            outputs=[prediction, feature_importance_plot, prediction_probability_plot]
        )
       
    
    return demo

# Launch the improved interface
if __name__ == "__main__":
    improved_interface = create_improved_interface()
    improved_interface.launch(share=True)


