import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Fill in the medical metrics below to predict the likelihood of diabetes.")

# ---------------------------------------------------------
# 1. Dataset & Model Training
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    """
    Train model using dataset. Replace 'diabetes_data.csv' with 
    the actual file path or URL if hosting dataset on GitHub.
    """
    try:
        df = pd.read_csv("1000rows_diabetes_prediction_small.csv")
    except FileNotFoundError:
        # Fallback dummy data structure matching your notebook columns
        data = {
            'gender': np.random.choice(['Female', 'Male'], 1000),
            'age': np.random.uniform(1, 80, 1000),
            'hypertension': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
            'heart_disease': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
            'smoking_history': np.random.choice(['No Info', 'former', 'never', 'not current', 'current', 'ever'], 1000),
            'bmi': np.random.uniform(10, 50, 1000),
            'HbA1c_level': np.random.uniform(3.5, 9.0, 1000),
            'blood_glucose_level': np.random.randint(80, 300, 1000),
            'diabetes': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
        }
        df = pd.DataFrame(data)

    # Clean duplicates
    df = df.drop_duplicates()

    # Features and Target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # One-hot encoding categorical variables
    X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_columns = train_model()

# ---------------------------------------------------------
# 2. User Input Form
# ---------------------------------------------------------
st.header("Patient Data Input")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "not current", "ever", "No Info"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120, step=1)

# ---------------------------------------------------------
# 3. Prediction Logic
# ---------------------------------------------------------
if st.button("Predict Risk", type="primary"):
    # Build single-row DataFrame from input
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }])

    # One-hot encode inputs
    input_encoded = pd.get_dummies(input_data, columns=['gender', 'smoking_history'])

    # Reindex to match training column structure (filling missing columns with 0)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** (Probability: {prediction_proba * 100:.1f}%)")
    else:
        st.success(f"✅ **Low Risk of Diabetes** (Probability: {prediction_proba * 100:.1f}%)")
