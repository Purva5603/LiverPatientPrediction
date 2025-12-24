import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction (Logistic Regression)",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Liver Disease Prediction System")
st.write(
    "This application uses Logistic Regression to assist healthcare professionals "
    "in identifying patients at risk of liver disease."
)

# -------------------------------
# Load Model Artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("lr_streamlit_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
gender_mapping = artifacts["gender_mapping"]
feature_order = artifacts["feature_order"]
threshold = artifacts["threshold"]

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0)
    direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase (U/L)", value=120.0)

with col2:
    alt = st.number_input("Alamine Aminotransferase (ALT)", value=30.0)
    ast = st.number_input("Aspartate Aminotransferase (AST)", value=35.0)
    total_proteins = st.number_input("Total Proteins (g/dL)", value=6.8)
    albumin = st.number_input("Albumin (g/dL)", value=4.0)
    agr = st.number_input("Albumin Globulin Ratio", value=1.3)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    # Create input DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_mapping[gender],
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alkaline_phosphotase,
        "Alamine_Aminotransferase": alt,
        "Aspartate_Aminotransferase": ast,
        "Total_Proteins": total_proteins,
        "Albumin": albumin,
        "Albumin_Globulin_Ratio": agr
    }])

    # Ensure correct column order
    input_df = input_df[feature_order]

    # 🔹 Scale input
    input_scaled = scaler.transform(input_df)

    # 🔹 Predict probability
    probability = model.predict_proba(input_scaled)[:, 1][0]

    # 🔹 Apply custom threshold
    prediction = 1 if probability >= threshold else 0

    # -------------------------------
    # Display Result
    # -------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Liver Disease**\n\n"
            f"Predicted Probability: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ **Low Risk of Liver Disease**\n\n"
            f"Predicted Probability: **{probability:.2f}**"
        )

    st.caption(
        "⚠️ This tool supports clinical decisions and should not replace professional medical diagnosis."
    )
