import streamlit as st

st.title('🚑 Random_Forest Model For Predicting Diabites 🚑')

st.write('Build By Ali Hamza and thanks to Sir Zafer for teching us how to make app like this')
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction App", page_icon="🩺", layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write(
    "Fill in the medical metrics below to predict the likelihood of diabetes."
)


# ---------------------------------------------------------
# 1. Load Pre-trained Model
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    # Load the model directly from your uploaded joblib file
    model = joblib.load("random_forest_model (1).joblib")
    return model


model = load_trained_model()

# Extract feature names the model expects (automatically saved by scikit-learn)
if hasattr(model, "feature_names_in_"):
    feature_columns = model.feature_names_in_
else:
    # Fallback feature list matching standard dummy encoding
    feature_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Male",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current",
    ]

# ---------------------------------------------------------
# 2. User Input Form
# ---------------------------------------------------------
st.header("Patient Data Input")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input(
        "Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0
    )
    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "not current", "ever", "No Info"],
    )
    bmi = st.number_input(
        "BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1
    )
    hba1c = st.number_input(
        "HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1
    )
    glucose = st.number_input(
        "Blood Glucose Level", min_value=50, max_value=400, value=120, step=1
    )

# ---------------------------------------------------------
# 3. Prediction Logic
# ---------------------------------------------------------
if st.button("Predict Risk", type="primary"):
    # Create input DataFrame
    input_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }
        ]
    )

    # One-hot encode inputs
    input_encoded = pd.get_dummies(
        input_data, columns=["gender", "smoking_history"]
    )

    # Reindex columns to strictly match the model's expected features
    input_encoded = input_encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    # Predict using the loaded joblib model
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Diabetes** (Probability:"
            f" {prediction_proba * 100:.1f}%)"
        )
    else:
        st.success(
            f"✅ **Low Risk of Diabetes** (Probability:"
            f" {prediction_proba * 100:.1f}%)"
        )
