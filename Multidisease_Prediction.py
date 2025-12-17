import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Disease Prediction System", layout="centered")

# Load models
kidney_model = pickle.load(open("kidney.pkl", "rb"))
liver_model = pickle.load(open("liver.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons.pkl", "rb"))

# Sidebar
st.sidebar.title("Disease Prediction")
disease = st.sidebar.selectbox(
    "Select Prediction Type",
    ("Kidney Disease", "Liver Disease", "Parkinson's Disease")
)

st.title("Multi Disease Prediction System")

# ---------------- KIDNEY ---------------- #

if disease == "Kidney Disease":
    st.subheader("Kidney Disease Prediction")

    hemo = st.number_input("Hemoglobin", format="%.2f")
    pcv = st.number_input("Packed Cell Volume")
    age = st.number_input("Age", 0, 100)
    sg = st.number_input("Specific Gravity", format="%.3f")
    sc = st.number_input("Serum Creatinine", format="%.2f")
    rbc = st.number_input("Red Blood Cells (encoded)", 0, 1)
    pc = st.number_input("Pus Cell (encoded)", 0, 1)
    wc = st.number_input("White Blood Cell Count")
    bgr = st.number_input("Blood Glucose Random")

    if st.button("Predict Kidney Disease"):
        input_data = np.array([[hemo, pcv, age, sg, sc, rbc, pc, wc, bgr]])
        result = kidney_model.predict(input_data)

        if result[0] == 1:
            st.error("Kidney Disease Detected")
        else:
            st.success("No Kidney Disease Detected")

# ---------------- LIVER ---------------- #

elif disease == "Liver Disease":
    st.subheader("Liver Disease Prediction")

    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    age = st.number_input("Age", 0, 100)
    direct_bilirubin = st.number_input("Direct Bilirubin", format="%.2f")
    total_bilirubin = st.number_input("Total Bilirubin", format="%.2f")

    if st.button("Predict Liver Disease"):
        input_data = np.array([[alkaline_phosphotase,
                                aspartate_aminotransferase,
                                alamine_aminotransferase,
                                age,
                                direct_bilirubin,
                                total_bilirubin]])
        result = liver_model.predict(input_data)

        if result[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("No Liver Disease Detected")

# ---------------- PARKINSONS ---------------- #

elif disease == "Parkinson's Disease":
    st.subheader("Parkinson's Disease Prediction")

    ppe = st.number_input("PPE", format="%.4f")
    spread1 = st.number_input("Spread1", format="%.4f")
    mdvp_apq = st.number_input("MDVP:APQ", format="%.4f")
    jitter_ddp = st.number_input("Jitter:DDP", format="%.4f")
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", format="%.3f")
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", format="%.3f")

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([[ppe, spread1, mdvp_apq,
                                jitter_ddp, mdvp_fhi, mdvp_fo]])
        result = parkinsons_model.predict(input_data)

        if result[0] == 1:
            st.error("Parkinson's Disease Detected")
        else:
            st.success("No Parkinson's Disease Detected")
