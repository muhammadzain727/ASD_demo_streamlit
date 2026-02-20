import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from utils import explain_prediction

st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Autism Detection"])

if page == "🏠 Home":

    st.markdown("""
    # 🧠 Autism Spectrum Disorder Detection in Children through Facial Images
    ### *Deep Learning Based Facial Pattern Analysis*
    """)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎓 University of Tartu

        ### 📘 Course
        **Machine Learning**

        ### 👨‍🏫 Course Instructor
        **Dmytro Fishman**

        ### 📌 Project Overview
        This project applies **InceptionV3 Deep Learning Architecture**
        with **Explainable AI (Grad-CAM)** to visualize facial regions
        influencing autism-related predictions.

        ⚠️ **Disclaimer:**  
        This system is for **academic and research purposes only**  
        and **must not** be considered a medical diagnostic tool.
        """)

    with col2:
        st.markdown("""
        ### 👨‍💻 Team Members
        - Muhammad Zain
        - Anet Lello	
        - Andrius Matšenas 
        - Muhammad Haris Irfan
        
        """)

    st.success("Use the sidebar to start Autism Detection →")

elif page == "🔍 Autism Detection":

    st.markdown("""
    # 🔍 Autism Detection
    Upload a child facial image to view prediction and explainability.
    """)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        label, prob, cam_image = explain_prediction("temp.jpg")

        TARGET_W, TARGET_H = 350, 350

        # Resize original image
        orig_img = Image.open(uploaded_file).convert("RGB")
        orig_img = np.array(orig_img)
        orig_img = cv2.resize(orig_img, (TARGET_W, TARGET_H))

        # Resize Grad-CAM image
        cam_img = cv2.resize(cam_image, (TARGET_W, TARGET_H))

        col1, col2 = st.columns(2)

        with col1:
            st.image(orig_img, caption="Original Image", width=TARGET_W)

        with col2:
            st.image(cam_img, caption="Model Attention (Grad-CAM)", width=TARGET_W)

        st.markdown(f"""
        ### 🧾 Prediction Result
        **Prediction:** `{label}`  
        **Confidence:** `{prob * 100:.2f}%`
        """)

        # 🧹 Automatically delete temp image
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")

    st.info("Explainable AI highlights regions influencing the model decision.")
