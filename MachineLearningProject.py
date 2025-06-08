import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and artifacts
@st.cache_resource
def load_model():
    model = joblib.load('lung_cancer_model.pkl')
    features = joblib.load('model_features.pkl')
    label_mapping = joblib.load('label_mapping.pkl')
    return model, features, label_mapping

model, features, label_mapping = load_model()

# Reverse mapping for display
risk_levels = {v: k for k, v in label_mapping.items()}

# App UI
#Sst.set_page_config(page_title="Lung Cancer Risk Predictor", layout="wide")

st.title("🫁 Lung Cancer Risk Prediction")
st.markdown("""
Predict the risk level (Low/Medium/High) based on patient characteristics.
""")

# Input form
with st.form("patient_form"):
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 10, 100, 50)
        air_pollution = st.slider("Air Pollution Exposure (1-8)", 1, 8, 3)
        alcohol = st.slider("Alcohol Consumption (1-8)", 1, 8, 2)
        smoking = st.slider("Smoking (1-8)", 1, 8, 3)
        
    with col2:
        genetic_risk = st.slider("Genetic Risk (1-7)", 1, 7, 3)
        chronic_lung = st.slider("Chronic Lung Disease (1-7)", 1, 7, 1)
        balanced_diet = st.slider("Balanced Diet (1-7)", 1, 7, 4)
        gender = st.radio("Gender", ["1", "2"], format_func=lambda x: "Male" if x == "1" else "Female")
    
    submitted = st.form_submit_button("Predict Risk Level")

if submitted:
    # Calculate total risk
    total_risk = air_pollution + alcohol + smoking
    
    # Create input dataframe
    input_data = {
        'Air Pollution': [air_pollution],
        'Alcohol use': [alcohol],
        'Smoking': [smoking],
        'Total_Risk': [total_risk],
        'Genetic Risk': [genetic_risk],
        'chronic Lung Disease': [chronic_lung],
        'Balanced Diet': [balanced_diet],
        'Gender_1': [1 if gender == "1" else 0],
        'Gender_2': [1 if gender == "2" else 0],
        'Age_Group_Young': [1 if age <= 30 else 0],
        'Age_Group_Middle': [1 if 30 < age <= 50 else 0],
        'Age_Group_Senior': [1 if age > 50 else 0]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Ensure column order matches training
    input_df = input_df[features]
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        st.subheader("Prediction Results")
        
        # Display risk level
        risk_level = risk_levels[prediction]
        if prediction == 2:  # High risk
            st.error(f"🚨 Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
        elif prediction == 1:  # Medium risk
            st.warning(f"⚠️ Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
        else:  # Low risk
            st.success(f"✅ Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
        
        # Show probability distribution
        st.write("Probability Distribution:")
        prob_df = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Risk Level'))
        
        # Show key factors
        st.subheader("Key Contributing Factors")
        feat_importance = pd.Series(model.feature_importances_, index=features)
        top_features = feat_importance.nlargest(5)
        
        for feat, importance in top_features.items():
            readable_feat = feat.replace('_', ' ').title()
            st.write(f"- {readable_feat}: {importance:.2f}")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


st.sidebar.header("About")
st.sidebar.markdown("""
This app predicts lung cancer risk using:
- Environmental factors (air pollution)
- Lifestyle choices (smoking, alcohol)
- Medical history (genetic risk, lung disease)
""")
st.sidebar.markdown("""
**Risk Level Explanation:**
- Low: 0-30% probability
- Medium: 30-70% probability  
- High: 70-100% probability
""")


