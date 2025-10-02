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
st.set_page_config(page_title="RH | EMOTION DETECTOR", layout="wide", page_icon="😊")

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
    <style>
    /* Background with animated gradient */
    body {
        background: linear-gradient(270deg, #f8f9fa, #ecf0f1, #dfe6e9);
        background-size: 600% 600%;
        animation: gradientBG 12s ease infinite;
        font-family: 'Segoe UI', sans-serif;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
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
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    /* Title & Subtitle */
    .title {
        text-align: center;
        font-size: 42px;
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
    /* Result Box */
    .result-box {
        margin-top: 20px;
        padding: 20px;
        border-radius: 15px;
        background: rgba(52, 152, 219, 0.15);
        border: 1px solid #3498db;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        font-size: 22px;
        animation: fadeIn 1.5s;
    }
    /* Alert Box for No Face Detected */
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }
    .alert-box {
        background: #f39c12;
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        animation: bounce 1s ease infinite;
        font-size: 18px;
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
st.markdown("<div class='navbar'>Face Emotion Detector | See Your Emotion & Enjoy !</div>", unsafe_allow_html=True)

# ------------------------------
# Sidebar Info & Social Links
# ------------------------------
st.sidebar.title("ℹ️ About App")
st.sidebar.info("This is an **AI-Powered Face Emotion Detector**. Detect emotions such as Happy, Sad, Angry, Neutral, and Surprise using Deep Learning.")
st.sidebar.success("👨‍💻 Developed by Rayhan Hussain")
st.sidebar.markdown("---")
st.sidebar.write("📌 **Tips:**\n- Use a clear photo\n- Good lighting helps\n- Try smiling 😉")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Connect with Me:**", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style="text-align: center; font-size: 16px;">
        <a href="https://www.facebook.com/Rayhanhcse" target="_blank" style="margin: 5px; text-decoration: none;"> 
            <i class="fab fa-facebook-square"></i> Facebook
        </a><br>
        <a href="https://www.instagram.com/Rayhanhcse" target="_blank" style="margin: 5px; text-decoration: none;"> 
            <i class="fab fa-instagram"></i> Instagram
        </a><br>
        <a href="https://www.linkedin.com/in/Rayhanhcse" target="_blank" style="margin: 5px; text-decoration: none;"> 
            <i class="fab fa-linkedin"></i> LinkedIn
        </a><br>
        <a href="https://github.com/Rayhanhcse" target="_blank" style="margin: 5px; text-decoration: none;"> 
            <i class="fab fa-github-square"></i> GitHub
        </a>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
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
emoji_map = {
    "Angry": "😠",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}
quotes = {
    "Angry": "Take a deep breath. Anger doesn't solve problems. 🌿",
    "Happy": "Keep smiling, happiness is contagious! 🌸",
    "Neutral": "Stay calm and balanced. ⚖️",
    "Sad": "Every storm passes. Brighter days are ahead. 🌤️",
    "Surprise": "Life is full of surprises, embrace them! 🎉"
}

# ------------------------------
# App Title
# ------------------------------
st.markdown("<div class='title'>Welcome to Emotion World!</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered | By Rayhan Hussain</div>", unsafe_allow_html=True)

# ------------------------------
# Dropdown for Mode Selection
# ------------------------------
mode = st.selectbox("🎯 Choose Mode", ["Webcam", "Upload Image"])

# ------------------------------
# Detect & Predict Function
# ------------------------------
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_emotion = None
    confidence = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        detected_emotion = emotion_labels[idx]
        confidence = float(np.max(prediction) * 100)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (52, 152, 219), 2)
        cv2.putText(frame, detected_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (231, 76, 60), 2)
    return frame, detected_emotion, confidence

# ------------------------------
# Emotion History
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Webcam Mode
# ------------------------------
if mode == "Webcam":
    st.info("📸 Use your webcam to capture a photo")
    uploaded_image = st.camera_input("Click below to capture your face 👇")
    if uploaded_image:
        with st.spinner("🔍 Analyzing emotions..."):
            time.sleep(1.5)
            image = Image.open(uploaded_image)
            frame = np.array(image.convert('RGB'))
            processed, emotion, confidence = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

# ------------------------------
# Upload Mode
# ------------------------------
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload your image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with st.spinner("🔍 Detecting emotions..."):
            time.sleep(1.5)
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed, emotion, confidence = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

# ------------------------------
# Show Recent Emotions
# ------------------------------
if st.session_state.history:
    st.subheader("🕒 Recent Emotions")
    st.write(" → ".join(st.session_state.history))

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
    <div class="footer">
      Copyright © 2025 | Rayhan Hussain - All Rights Reserved
    </div>
""", unsafe_allow_html=True)

