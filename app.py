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
st.markdown("""
<style>
/* ---------------------- Body & Background ---------------------- */
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(270deg, #f8f9fa, #ecf0f1, #dfe6e9);
    background-size: 600% 600%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* ---------------------- Navbar ---------------------- */
.navbar {
    width: 100%;
    padding: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #fff;
    border-radius: 0 0 25px 25px;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 400% 400%;
    animation: gradientAnimation 20s ease infinite;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    text-shadow: 2px 3px 8px rgba(0,0,0,0.3);
}
@keyframes gradientAnimation {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* ---------------------- Titles ---------------------- */
.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#2c3e50;
    margin-top:25px;
    text-shadow: 2px 3px 12px rgba(0,0,0,0.2);
    animation: fadeIn 2s;
}
.subtitle {
    text-align:center;
    font-size:22px;
    color:#555;
    margin-bottom:35px;
    animation: fadeIn 2.5s;
}

/* ---------------------- Result Box ---------------------- */
.result-box {
    margin-top:20px;
    padding:25px;
    border-radius:20px;
    background: rgba(52,152,219,0.25);
    border:1px solid #3498db;
    color:#2c3e50;
    font-weight:bold;
    text-align:center;
    font-size:24px;
    animation: fadeIn 1.5s;
    box-shadow: 0px 10px 25px rgba(52,152,219,0.5);
    backdrop-filter: blur(8px);
}

/* ---------------------- Alert Box ---------------------- */
.alert-box {
    background: linear-gradient(45deg,#f39c12,#e67e22);
    color:white;
    font-weight:bold;
    padding:18px;
    border-radius:18px;
    text-align:center;
    margin-top:20px;
    animation: bounce 1.5s infinite;
    font-size:20px;
    box-shadow: 0px 6px 20px rgba(243,156,18,0.6);
}
@keyframes bounce { 0%,20%,50%,80%,100%{transform:translateY(0);} 40%{transform:translateY(-12px);} 60%{transform:translateY(-6px);} }

/* ---------------------- Sidebar Social ---------------------- */
.sidebar .stMarkdown a {
    display:block;
    margin:6px 0;
    padding:10px 15px;
    border-radius:14px;
    text-decoration:none;
    color:white !important;
    font-weight:bold;
    text-align:center;
    transition: all 0.3s ease;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.3);
}
.sidebar .stMarkdown a:hover {
    transform: scale(1.08);
    box-shadow: 0px 12px 28px rgba(0,0,0,0.45);
    filter: brightness(1.2);
}

/* ---------------------- Mode Buttons ---------------------- */
.mode-buttons { display:flex; justify-content:center; gap:25px; margin:30px 0; }
.mode-btn {
    padding:16px 32px;
    font-size:20px;
    font-weight:bold;
    border-radius:18px;
    border:none;
    cursor:pointer;
    transition: all 0.4s ease;
    color:white;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 300% 300%;
    animation: gradientMove 6s ease infinite;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}
.mode-btn:hover {
    transform: scale(1.12);
    box-shadow: 0px 12px 30px rgba(0,0,0,0.5);
}
.mode-btn.active {
    border: 4px solid #fff;
    transform: scale(1.15);
    box-shadow: 0px 14px 35px rgba(0,0,0,0.55);
}
@keyframes gradientMove {
    0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;}
}

/* ---------------------- Footer ---------------------- */
.footer {
    position:fixed;
    left:0;
    bottom:0;
    width:100%;
    text-align:center;
    color:#333;
    font-size:15px;
    padding:16px;
    background: rgba(255,255,255,0.8);
    border-top:1px solid rgba(200,200,200,0.3);
    box-shadow:0px -6px 18px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
}
.footer:hover { background: rgba(255,255,255,0.95); }

/* ---------------------- Animations ---------------------- */
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
@keyframes fadeInDown { from {opacity:0; transform:translateY(-20px);} to {opacity:1; transform:translateY(0);} }

/* ---------------------- Responsive ---------------------- */
@media(max-width:768px) {
    .title{font-size:32px;}
    .subtitle{font-size:16px;}
    .mode-btn{padding:12px 24px; font-size:16px;}
    .navbar{font-size:20px;}
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
st.markdown("<div class='navbar'>Welcome to Emotion World!</div>", unsafe_allow_html=True)

# ------------------------------
# Sidebar with social links
st.sidebar.title("ℹ️ About App")
st.sidebar.info("This is an **AI-Powered Face Emotion Detector**. Detect emotions such as Happy, Sad, Angry, Neutral, and Surprise using Deep Learning.")
st.sidebar.markdown("---")
st.sidebar.write("📌 **Tips:**\n- Use a clear photo\n- Good lighting helps\n- Try smiling 😉")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Connect with Me:**", unsafe_allow_html=True)
st.sidebar.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<div style="display:flex; flex-direction:column; align-items:center;">
<a href="https://www.facebook.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:12px; background:#3b5998; color:white; text-decoration:none; width:160px; text-align:center;"><i class="fab fa-facebook-square"></i> Facebook</a>
<a href="https://www.instagram.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:12px; background:#E1306C; color:white; text-decoration:none; width:160px; text-align:center;"><i class="fab fa-instagram"></i> Instagram</a>
<a href="https://www.linkedin.com/in/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:12px; background:#0077B5; color:white; text-decoration:none; width:160px; text-align:center;"><i class="fab fa-linkedin"></i> LinkedIn</a>
<a href="https://github.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:12px; background:#333; color:white; text-decoration:none; width:160px; text-align:center;"><i class="fab fa-github-square"></i> GitHub</a>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
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
# Mode Buttons
if "mode" not in st.session_state: st.session_state.mode = None

col1, col2 = st.columns(2)
with col1:
    if st.button("📸 Use Webcam"):
        st.session_state.mode = "Webcam"
        st.experimental_rerun()
with col2:
    if st.button("📂 Upload Image"):
        st.session_state.mode = "Upload Image"
        st.experimental_rerun()

# Highlight active button
if st.session_state.mode:
    active_css = f"""
    <style>
    .stButton button[title="{st.session_state.mode}"] {{
        border: 4px solid white !important;
        transform: scale(1.1) !important;
        box-shadow:0px 12px 28px rgba(255,255,255,0.6);
    }}
    </style>
    """
    st.markdown(active_css, unsafe_allow_html=True)

# ------------------------------
# Detect & Predict
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    detected_emotion = None
    confidence = None
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi,(48,48))
        roi = roi.reshape(1,48,48,1)/255.0
        prediction = model.predict(roi,verbose=0)
        idx = np.argmax(prediction)
        detected_emotion = emotion_labels[idx]
        confidence = float(np.max(prediction)*100)
        # Neon glowing rectangle
        color_map = {"Angry":(255,50,50),"Happy":(50,255,50),"Neutral":(200,200,200),"Sad":(50,50,255),"Surprise":(255,255,0)}
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_map.get(detected_emotion,(255,255,255)),3)
        cv2.putText(frame, detected_emotion,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    return frame, detected_emotion, confidence

# ------------------------------
# Emotion History
if "history" not in st.session_state: st.session_state.history=[]

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
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history)>5: st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

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
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history)>5: st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

# ------------------------------
# Recent Emotions (horizontal carousel)
if st.session_state.history:
    st.subheader("🕒 Recent Emotions")
    st.markdown(
        "<div style='display:flex; overflow-x:auto; gap:10px; padding:10px;'>"
        + "".join([f"<div style='min-width:100px; padding:10px; border-radius:15px; background:rgba(52,152,219,0.25); text-align:center;'>{h}</div>" for h in st.session_state.history])
        + "</div>",
        unsafe_allow_html=True
    )

# ------------------------------
# Footer
st.markdown("<div class='footer'>Copyright © 2025 | Rayhan Hussain - All Rights Reserved</div>", unsafe_allow_html=True)
