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
    padding: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    border-radius: 0 0 15px 15px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite, fadeInDown 1.5s;
}
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
/* Title & Subtitle */
.title { text-align: center; font-size: 42px; font-weight: bold; color: #2c3e50; margin-top: 25px; animation: fadeIn 2s; }
.subtitle { text-align: center; font-size: 18px; color: #444; margin-bottom: 35px; animation: fadeIn 2.5s; }
/* Mode Buttons */
.mode-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 40px;
    margin: 40px 0;
}
.mode-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.8);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
    width: 260px;
}
.mode-box:hover {
    transform: translateY(-5px);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
}
.mode-btn {
    padding: 14px 28px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    color: white;
    margin-top: 15px;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 300% 300%;
    animation: gradientMove 6s ease infinite;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}
.mode-btn:hover {
    transform: scale(1.05);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}
.mode-btn.active {
    border: 3px solid #fff;
    transform: scale(1.08);
    box-shadow: 0px 12px 28px rgba(0,0,0,0.4);
}
.mode-text {
    font-size: 15px;
    margin-top: 10px;
    color: #333;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
/* Footer */
.footer { position: fixed; left: 0; bottom: 0; width: 100%; text-align: center; color:#666; font-size:13px; padding:12px; background: rgba(250,250,250,0.5); border-top:1px solid rgba(200,200,200,0.3); box-shadow:0px -4px 12px rgba(0,0,0,0.1); backdrop-filter: blur(6px); }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
st.markdown("<div class='navbar'>Welcome to Emotion World!</div>", unsafe_allow_html=True)

# ------------------------------
# Sidebar
st.sidebar.title("ℹ️ About App")
st.sidebar.info("This is an **AI-Powered Face Emotion Detector**. Detect emotions such as Happy, Sad, Angry, Neutral, and Surprise using Deep Learning.")
st.sidebar.success("👨‍💻 Developed by Rayhan Hussain")

# ------------------------------
# Load Haar Cascade
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)
face_cascade = cv2.CascadeClassifier(cascade_file)

# ------------------------------
# Load Model
model = load_model("emotion_new.h5", compile=False)
emotion_labels = ['Angry','Happy','Neutral','Sad','Surprise']
emoji_map = {"Angry":"😠","Happy":"😊","Neutral":"😐","Sad":"😢","Surprise":"😲"}
quotes = {
    "Angry":"Take a deep breath. Anger doesn't solve problems. 🌿",
    "Happy":"Keep smiling, happiness is contagious! 🌸",
    "Neutral":"Stay calm and balanced. ⚖️",
    "Sad":"Every storm passes. Brighter days are ahead. 🌤️",
    "Surprise":"Life is full of surprises, embrace them! 🎉"
}

# ------------------------------
# App Title
st.markdown("<div class='title'>Face Emotion Detector | See Your Emotion & Enjoy</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered | By Rayhan Hussain</div>", unsafe_allow_html=True)

# ------------------------------
# Mode Buttons with Description
if "mode" not in st.session_state:
    st.session_state.mode = None

st.markdown('<div class="mode-container">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="mode-box">', unsafe_allow_html=True)
    if st.button("📸 Use Webcam", key="webcam_btn"):
        st.session_state.mode = "Webcam"
    st.markdown("<div class='mode-text'>Capture your face live using your webcam for real-time emotion detection.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="mode-box">', unsafe_allow_html=True)
    if st.button("📂 Upload Image", key="upload_btn"):
        st.session_state.mode = "Upload Image"
    st.markdown("<div class='mode-text'>Upload a photo (jpg, jpeg, png) to detect your facial emotion instantly.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Detect & Predict
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    detected_emotion, confidence = None, None
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi,(48,48))
        roi = roi.reshape(1,48,48,1)/255.0
        prediction = model.predict(roi,verbose=0)
        idx = np.argmax(prediction)
        detected_emotion = emotion_labels[idx]
        confidence = float(np.max(prediction)*100)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(52,152,219),2)
        cv2.putText(frame, detected_emotion,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(231,76,60),2)
    return frame, detected_emotion, confidence

# ------------------------------
# Emotion History
if "history" not in st.session_state: 
    st.session_state.history=[]

# ------------------------------
# Webcam Mode
if st.session_state.mode == "Webcam":
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
                st.success(f"{emoji_map[emotion]} Detected Emotion: {emotion} ({confidence:.2f}%)")
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history)>5: st.session_state.history.pop(0)
            else:
                st.error("⚠️ No face detected. Try again with better lighting!")

# ------------------------------
# Upload Mode
elif st.session_state.mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload your image (jpg,jpeg,png)", type=["jpg","jpeg","png"])
    if uploaded_file:
        with st.spinner("🔍 Detecting emotions..."):
            time.sleep(1.5)
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed, emotion, confidence = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.success(f"{emoji_map[emotion]} Detected Emotion: {emotion} ({confidence:.2f}%)")
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history)>5: st.session_state.history.pop(0)
            else:
                st.error("⚠️ No face detected. Try again with better lighting!")

# ------------------------------
# Recent Emotions
if st.session_state.history:
    st.subheader("🕒 Recent Emotions")
    st.write(" → ".join(st.session_state.history))

# ------------------------------
# Footer
st.markdown("""
<div class="footer">Copyright © 2025 | Rayhan Hussain - All Rights Reserved</div>
""", unsafe_allow_html=True)
