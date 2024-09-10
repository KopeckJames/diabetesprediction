# Diabetes Prediction

## Project 2 - Group 2

### Team Members:  
- Luc  
- Aaron  
- Aaliyah  
- Ryan  
- James

https://huggingface.co/spaces/kopeck/diabetespredictiontool​
  
# About the Dataset

The **Diabetes Health Dataset** is a comprehensive collection of health and lifestyle information gathered from patients, with the aim of understanding and predicting diabetes. The dataset includes various features such as age, gender, health habits (smoking, alcohol consumption, exercise), and medical history. The goal is to uncover patterns in these factors that can help identify the likelihood of diabetes.

## Data Source

This dataset was compiled from patient health records specifically focusing on diabetes monitoring. It has been thoroughly cleaned and anonymized to ensure the privacy of individuals, with no personal identifiers (such as names or addresses) included. The dataset was uploaded to Kaggle by **Rabie El Kharoua** to facilitate research into diabetes.

## Dataset Overview

The dataset contains **46 columns** (features) and thousands of patient records (rows). Each row represents the health profile of one patient, capturing personal and medical information. Below is a breakdown of the key feature categories:

### 1. Personal Information:
- **Patient ID**: Unique identifier for each individual.
- **Age**: The age of the patient.
- **Gender**: Encoded as 0 for male and 1 for female.
- **Ethnicity**: The ethnic background of the patient.
- **Socioeconomic Status**: A measure of the patient’s financial condition.
- **Education Level**: The highest level of education attained by the patient.

### 2. Health Habits:
- **BMI**: Body Mass Index, indicating whether the patient is underweight, normal weight, or overweight.
- **Smoking**: Binary indicator (1 for smokers, 0 for non-smokers).
- **Alcohol Consumption**: The patient’s frequency or amount of alcohol intake.
- **Physical Activity**: The level of regular exercise performed by the patient.

### 3. Medical History:
- **Tingling Hands/Feet**: Binary indicator for tingling sensations in extremities.
- **Exposure to Chemicals/Metals**: Indicator for past exposure to harmful substances.
- **Water Quality**: Assesses the quality of water consumed.
- **Medical Checkups**: Frequency of medical visits for routine checkups.

### 4. Diabetes Information:
- **Diagnosis**: Binary indicator (1 for diabetes diagnosis, 0 for no diagnosis).
- **Medication Adherence**: Degree to which the patient follows prescribed medication.
- **Health Literacy**: Patient's understanding of their own health and medical care.

## Applications of the Dataset

This dataset is ideal for various analyses and research purposes:
- **Predictive Analytics**: Build machine learning models to predict diabetes risk based on health habits and medical history.
- **Behavioral Insights**: Analyze patient behaviors such as medication adherence and health literacy.
- **Data Visualization**: Create visual reports highlighting trends and correlations in diabetes risk factors, such as age or exercise levels.

## Importance of This Dataset

The **Diabetes Health Dataset** serves as an important resource for anyone studying diabetes. It can be used to:
- Pinpoint key factors that influence the prevention and management of diabetes.
- Support the development of healthcare programs focused on diabetic patients or individuals at high risk.

## Model Performance

In this project, we tested various machine learning models to predict diabetes. Among the models evaluated, **CatBoost** demonstrated the highest accuracy and produced the best confusion matrix scores. Below is a list of models tested:

- `LogisticRegression`: LogisticRegression()
- `SVC`: Support Vector Classifier (SVC)
- `DecisionTree`: DecisionTreeClassifier()
- `RandomForest`: RandomForestClassifier()
- `ExtraTrees`: ExtraTreesClassifier()
- `AdaBoost`: AdaBoostClassifier()
- `GradientBoost`: GradientBoostingClassifier()
- `XGBoost`: XGBClassifier(use_label_encoder=False, eval_metric='logloss')
- `LightGBM`: LGBMClassifier()
- `CatBoost`: CatBoostClassifier()

**CatBoost** achieved the best overall performance, making it the top choice for our predictive modeling.








# Project Requirements

## Data Model Implementation (25 points)
* **Data Extraction, Cleaning, and Transformation**:  
   There is a Jupyter notebook that thoroughly describes the process of data extraction, cleaning, and transformation. The cleaned data is exported as CSV files for the machine learning model.  
   **(10 points)**

* **Model Initialization and Training**:  
   A Python script initializes, trains, and evaluates a model, or alternatively, loads a pretrained model.  
   **(10 points)**

* **Model Accuracy**:  
   The model demonstrates meaningful predictive power with at least 75% classification accuracy or an R-squared value of 0.80.  
   **(5 points)**

## Data Model Optimization (25 points)
* **Documentation of Model Optimization**:  
   The optimization and evaluation process is documented, showing the iterative changes made to the model and the resulting impact on performance. This documentation is presented either in a CSV/Excel table or directly in the Python script.  
   **(15 points)**

* **Overall Model Performance**:  
   At the end of the script, the overall model performance is printed or displayed.  
   **(10 points)**

## GitHub Documentation (25 points)
* **Repository Cleanliness**:  
   The GitHub repository is organized, free of unnecessary files and folders, and has an appropriate `.gitignore` in use.  
   **(10 points)**

* **Polished README**:  
   The README file is customized and serves as a polished presentation of the project’s content.  
   **(15 points)**

## Presentation Requirements (25 points)
Your presentation should include the following:

* **Executive Summary**:  
   Provide an overview of the project and its goals.  
   **(5 points)**

* **Data Collection, Cleanup, and Exploration**:  
   Describe the processes involved in collecting, cleaning, and exploring the data.  
   **(5 points)**

* **Project Approach**:  
   Explain the approach your group took to achieve the project’s goals.  
   **(5 points)**

* **Future Research/Questions**:  
   Discuss any additional questions that arose, what your group would research next with more time, or outline plans for future development.  
   **(3 points)**

* **Results and Conclusions**:  
   Share the results and conclusions from the analysis or application.  
   **(3 points)**

* **Slide Quality**:  
   Ensure the slides effectively demonstrate the project and are visually clean and professional.  
   **(4 points total - 2 points for content demonstration, 2 points for cleanliness and professionalism)**
