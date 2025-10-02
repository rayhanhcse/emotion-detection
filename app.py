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
st.set_page_config(page_title="RH-FED | Emotion Detector", layout="wide")

# ------------------------------
# Custom CSS for Professional Look
# ------------------------------
st.markdown("""
    <style>
    body {
        background-color: #fdfdfd;
    }
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #2c3e50;
        animation: fadeInDown 1s;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
        animation: fadeIn 2s;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 12px;
        padding: 8px;
        background-color: #f9f9f9;
        border-top: 1px solid #eee;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

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
st.markdown("<div class='title'>😊 Face Emotion Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by Deep Learning | By Rayhan Hussain</div>", unsafe_allow_html=True)

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
# Webcam Mode
# ------------------------------
if mode == "Webcam":
    st.info("📸 Use your webcam and capture a photo")
    uploaded_image = st.camera_input("Click to capture")
    
    if uploaded_image:
        with st.spinner("Analyzing emotions..."):
            time.sleep(1)  # Animation effect
            image = Image.open(uploaded_image)
            frame = np.array(image.convert('RGB'))
            processed = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)

# ------------------------------
# Upload Mode
# ------------------------------
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner("Detecting emotions..."):
            time.sleep(1.2)  # Loading animation
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
    <div class="footer">
        © 2025 | Rayhan Hussain - All Rights Reserved
    </div>
""", unsafe_allow_html=True)
