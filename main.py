import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from datetime import date
import streamlit as st

# Load the dataset
df = pd.read_csv("F:\\Training\\ML\\CSV files\\medical data.csv")

# Preprocessing steps
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], format='%d-%m-%Y', errors='coerce')
median_date = df['DateOfBirth'].dropna().median()
df['DateOfBirth'].fillna(median_date, inplace=True)

categorical_columns = ['Name', 'Gender', 'Symptoms', 'Causes', 'Disease', 'Medicine']
for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

df = df.drop(columns=["Name"])

def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df['Age'] = df['DateOfBirth'].apply(calculate_age)
df = df.drop(columns=["DateOfBirth"])

df[['Symptoms_1', 'Symptoms_2']] = df['Symptoms'].str.split(', ', expand=True, n=1).fillna('None')
df[['Medicine_1', 'Medicine_2']] = df['Medicine'].str.split(', ', expand=True, n=1).fillna('None')
df = df.drop(columns=["Symptoms", "Medicine"])

# Combine unique symptoms
all_symptoms = pd.concat([df['Symptoms_1'], df['Symptoms_2']]).unique()

label_encoders = {}
for column in ['Gender', 'Causes', 'Disease', 'Symptoms_1', 'Symptoms_2', 'Medicine_1', 'Medicine_2']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Use combined symptoms for encoding
symptoms_encoder = LabelEncoder()
symptoms_encoder.fit(all_symptoms)

X = df.drop(columns=['Causes', 'Disease', 'Medicine_1', 'Medicine_2'])
y = df[['Causes', 'Disease', 'Medicine_1', 'Medicine_2']]

multi_target_rf = MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1)
multi_target_rf.fit(X, y)

def predict_cause_disease_medicine(age, gender, symptom1, symptom2):
    input_data = pd.DataFrame({
        'Gender': label_encoders['Gender'].transform([gender]),
        'Age': [age],
        'Symptoms_1': symptoms_encoder.transform([symptom1]),
        'Symptoms_2': symptoms_encoder.transform([symptom2])
    })
    prediction = multi_target_rf.predict(input_data)
    predicted_cause = label_encoders['Causes'].inverse_transform([prediction[0][0]])[0]
    predicted_disease = label_encoders['Disease'].inverse_transform([prediction[0][1]])[0]
    predicted_medicine_1 = label_encoders['Medicine_1'].inverse_transform([prediction[0][2]])[0]
    predicted_medicine_2 = label_encoders['Medicine_2'].inverse_transform([prediction[0][3]])[0]
    return predicted_cause, predicted_disease, predicted_medicine_1, predicted_medicine_2

# Streamlit App
st.title('Disease predictor and drug recommendation system')

age = st.number_input('Age', min_value=0, max_value=120, value=25, step=1)
gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
symptom1 = st.selectbox('First Symptom', symptoms_encoder.classes_)
symptom2 = st.selectbox('Second Symptom', symptoms_encoder.classes_)

if st.button('Predict'):
    try:
        predicted_cause, predicted_disease, predicted_medicine_1, predicted_medicine_2 = predict_cause_disease_medicine(
            age, gender, symptom1, symptom2)
        
        st.success('Prediction Results:')
        st.write(f'Cause: {predicted_cause}')
        st.write(f'Disease: {predicted_disease}')
        st.write(f'Medicine 1: {predicted_medicine_1}')
        st.write(f'Medicine 2: {predicted_medicine_2 if predicted_medicine_2 != "None" else "No additional medicine recommended"}')
    except Exception as e:
        st.error(f"Error occurred: {e}")
