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
st.set_page_config(page_title="RH-FED | Emotion Detector", layout="wide", page_icon="😊")

# ------------------------------
# Custom CSS for Modern UI
# ------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f9f9f9, #eef2f3);
        font-family: 'Segoe UI', sans-serif;
    }
    .navbar {
        width: 100%;
        padding: 12px;
        background: rgba(44, 62, 80, 0.95);
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        border-radius: 0 0 12px 12px;
        animation: fadeInDown 1s;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        animation: fadeIn 2s;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 25px;
        animation: fadeIn 2.5s;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 13px;
        padding: 10px;
        background: rgba(250, 250, 250, 0.95);
        border-top: 1px solid #ddd;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(135deg, #3498db, #2ecc71);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2980b9, #27ae60);
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
# ------------------------------
st.markdown("<div class='navbar'>RH-FED | Face Emotion Detection</div>", unsafe_allow_html=True)

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
st.markdown("<div class='title'>😊 Advanced Emotion Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time AI Powered Emotion Recognition</div>", unsafe_allow_html=True)

# ------------------------------
# Mode Selection
# ------------------------------
mode = st.selectbox("🎯 Choose Mode", ["Real-time Webcam", "Upload Image"])

# ------------------------------
# Function for Prediction
# ------------------------------
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 204, 113), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (231, 76, 60), 2)
    return frame

# ------------------------------
# Real-time Webcam Mode
# ------------------------------
if mode == "Real-time Webcam":
    st.warning("🔴 Real-time emotion detection requires webcam access")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = detect_and_predict(frame)
            FRAME_WINDOW.image(processed, channels="RGB", use_container_width=True)

        cap.release()

# ------------------------------
# Upload Mode
# ------------------------------
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner("Analyzing emotions..."):
            time.sleep(1.2)
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
    <div class="footer">
        © 2025 | RH-FED | Developed by Rayhan Hussain | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
