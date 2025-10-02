import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import urllib.request
import time

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="RH-FED", layout="wide", page_icon="😊")

# ------------------------------
# Custom CSS for Beautiful UI
# ------------------------------
st.markdown("""
    <style>
    /* Global Background */
    body {
        background: linear-gradient(135deg, #f8f9fa, #ecf0f1);
        font-family: 'Segoe UI', sans-serif;
    }
    /* Navbar */
    .navbar {
        width: 100%;
        padding: 18px;
        background: linear-gradient(90deg, #2c3e50, #3498db);
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border-radius: 0 0 15px 15px;
        animation: fadeInDown 1s;
    }
    /* Title & Subtitle */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 25px;
        animation: fadeIn 2s;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #444;
        margin-bottom: 35px;
        animation: fadeIn 2.5s;
    }
    /* Cards & Inputs */
    .stSelectbox, .stFileUploader, .stCameraInput {
        background: rgba(255,255,255,0.75) !important;
        padding: 15px;
        border-radius: 12px !important;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3498db, #2ecc71);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2980b9, #27ae60);
        transform: scale(1.05);
        cursor: pointer;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    /* Result Box */
    .result-box {
        margin-top: 20px;
        padding: 15px;
        border-radius: 12px;
        background: rgba(52, 152, 219, 0.1);
        border: 1px solid #3498db;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
        animation: fadeIn 1.5s;
    }
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #666;
        font-size: 13px;
        padding: 12px;
        background: rgba(250, 250, 250, 0.95);
        border-top: 1px solid #ddd;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Responsive */
    @media (max-width: 768px) {
        .title { font-size: 28px; }
        .subtitle { font-size: 14px; }
        .stButton>button { padding: 10px 20px; font-size: 14px; }
        .navbar { font-size: 18px; }
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
# ------------------------------
st.markdown("<div class='navbar'>🤖 RH-FED | AI Face Emotion Detector</div>", unsafe_allow_html=True)

# ------------------------------
# Load Haar Cascade
# ------------------------------
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)

face_cascade = cv2.CascadeClassifier(cascade_file)

# ------------------------------
# Load Model
# ------------------------------
model = load_model("emotion_new.h5", compile=False)
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------------
# App Title
# ------------------------------
st.markdown("<div class='title'>😊 Welcome to Emotion World!</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered | By Rayhan Hussain</div>", unsafe_allow_html=True)

# ------------------------------
# Dropdown for Mode Selection
# ------------------------------
mode = st.selectbox("🎯 Choose Mode", ["Webcam", "Upload Image"])

# ------------------------------
# Function for Prediction
# ------------------------------
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_emotion = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(roi, verbose=0)
        detected_emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (52, 152, 219), 2)
        cv2.putText(frame, detected_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (231, 76, 60), 2)
    return frame, detected_emotion

# ------------------------------
# Webcam Mode
# ------------------------------
if mode == "Webcam":
    st.info("📸 Use your webcam to capture a photo")
    uploaded_image = st.camera_input("Click below to capture your face 👇")
    
    if uploaded_image:
        with st.spinner("🔍 Analyzing emotions..."):
            time.sleep(1.2)
            image = Image.open(uploaded_image)
            frame = np.array(image.convert('RGB'))
            processed, emotion = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>😀 Detected Emotion: <b>{emotion}</b></div>", unsafe_allow_html=True)

# ------------------------------
# Upload Mode
# ------------------------------
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload your image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner("🔍 Detecting emotions..."):
            time.sleep(1.2)
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed, emotion = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>😀 Detected Emotion: <b>{emotion}</b></div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
    <div class="footer">
       Copyright © 2025 | Rayhan Hussain - All Rights Reserved
    </div>
""", unsafe_allow_html=True)


