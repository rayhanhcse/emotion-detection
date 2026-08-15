import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
    font-size: 26px;
    font-weight: bold;
    color: white;
    border-radius: 0 0 20px 20px;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite, fadeInDown 1.5s;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
    text-shadow: 1px 2px 6px rgba(0,0,0,0.3);
}
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title & Subtitle */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: bold;
    color: #2c3e50;
    margin-top: 25px;
    animation: fadeIn 2s;
    text-shadow: 2px 3px 8px rgba(0,0,0,0.15);
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 35px;
    animation: fadeIn 2.5s;
}

/* Result Box */
.result-box {
    margin-top: 20px;
    padding: 20px;
    border-radius: 15px;
    background: rgba(52,152,219,0.2);
    border:1px solid #3498db;
    color:#2c3e50;
    font-weight:bold;
    text-align:center;
    font-size:22px;
    animation: fadeIn 1.5s;
    box-shadow: 0px 8px 20px rgba(52,152,219,0.4);
    backdrop-filter: blur(6px);
}

/* Alert Box */
.alert-box {
    background: linear-gradient(45deg,#f39c12,#e67e22);
    color:white;
    font-weight:bold;
    padding:15px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
    animation: bounce 1.5s infinite;
    font-size:18px;
    box-shadow: 0px 6px 18px rgba(243,156,18,0.5);
}
@keyframes bounce {
    0%,20%,50%,80%,100%{transform:translateY(0);}
    40%{transform:translateY(-10px);}
    60%{transform:translateY(-5px);}
}

/* Sidebar social links */
.sidebar .stMarkdown a {
    display:block;
    margin:6px 0;
    padding:10px 15px;
    border-radius:12px;
    text-decoration:none;
    color:white !important;
    font-weight:bold;
    text-align:center;
    transition: all 0.3s ease;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.25);
}
.sidebar .stMarkdown a:hover {
    transform: scale(1.07);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    filter: brightness(1.1);
}

/* Mode Buttons */
.mode-buttons { display: flex; justify-content: center; gap: 20px; margin: 25px 0; }
.mode-btn {
    padding: 14px 28px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    color: white;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 300% 300%;
    animation: gradientMove 6s ease infinite;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
}
.mode-btn:hover {
    transform: scale(1.08);
    box-shadow: 0px 10px 28px rgba(0,0,0,0.35);
}
.mode-btn.active {
    border: 3px solid #fff;
    transform: scale(1.1);
    box-shadow: 0px 12px 30px rgba(0,0,0,0.45);
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color:#333;
    font-size:14px;
    padding:14px;
    background: rgba(255,255,255,0.7);
    border-top:1px solid rgba(200,200,200,0.3);
    box-shadow:0px -6px 15px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
}
.footer:hover {
    background: rgba(255,255,255,0.9);
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
<a href="https://www.facebook.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:10px; background:#3b5998; color:white; text-decoration:none; width:150px; text-align:center;"><i class="fab fa-facebook-square"></i> Facebook</a>
<a href="https://www.instagram.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:10px; background:#E1306C; color:white; text-decoration:none; width:150px; text-align:center;"><i class="fab fa-instagram"></i> Instagram</a>
<a href="https://www.linkedin.com/in/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:10px; background:#0077B5; color:white; text-decoration:none; width:150px; text-align:center;"><i class="fab fa-linkedin"></i> LinkedIn</a>
<a href="https://github.com/Rayhanhcse" target="_blank" style="margin:5px; padding:8px 12px; border-radius:10px; background:#333; color:white; text-decoration:none; width:150px; text-align:center;"><i class="fab fa-github-square"></i> GitHub</a>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.success("👨‍💻 Developed by Rayhan Hussain")

# ------------------------------
# Load Haar Cascade
# Uses the copy bundled inside opencv-python-headless instead of downloading it —
# avoids network dependency/rate limits on Streamlit Cloud. Make sure requirements.txt
# lists ONLY opencv-python-headless (not opencv-python / opencv-contrib-python too),
# otherwise you'll get "module 'cv2' has no attribute 'CascadeClassifier'".
cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_file)
if face_cascade.empty():
    st.error("Failed to load the Haar Cascade classifier. Check your OpenCV installation.")
    st.stop()

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
st.markdown("<div class='title'>AI-Based Facial Emotion Analysis  System Using Deep Learning | See Your Emotion & Enjoy</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered | By Rayhan Hussain</div>", unsafe_allow_html=True)

# ------------------------------
# Mode Buttons with text
if "mode" not in st.session_state:
    st.session_state.mode = None

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("👉 **Snapshot from your webcam**")
    if st.button("📸 Webcam Snapshot"):
        st.session_state.mode = "Webcam"
with col2:
    st.markdown("👉 **Upload an image from your device**")
    if st.button("📂 Upload Image Detection"):
        st.session_state.mode = "Upload Image"
with col3:
    st.markdown("👉 **Continuous live detection**")
    if st.button("🎥 Live Webcam (Real-Time)"):
        st.session_state.mode = "Live"

# ------------------------------
# Detect & Predict (with glowing box)
# Expects an RGB frame, returns (processed_rgb_frame, emotion, confidence)
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

        # Glowing rectangle
        for i in range(6, 0, -2):  
            overlay = frame.copy()
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(52,152,219),i+4)
            alpha = 0.25 * (i/6)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Solid border
        cv2.rectangle(frame,(x,y),(x+w,y+h),(52,152,219),2)

        # Shadow text + main text
        cv2.putText(frame, detected_emotion,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),4)
        cv2.putText(frame, detected_emotion,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(231,76,60),2)
    return frame, detected_emotion, confidence

# ------------------------------
# Emotion History
if "history" not in st.session_state: st.session_state.history=[]

# ------------------------------
# Webcam Snapshot Mode
if st.session_state.mode == "Webcam":
    st.info("📸 Turn on your webcam and capture your face")
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
# Live (Real-Time) Webcam Mode via streamlit-webrtc
elif st.session_state.mode == "Live":
    st.info("🎥 Allow camera access below for continuous live emotion detection.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.last_emotion = None
            self.last_confidence = None

        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            processed_rgb, emotion, confidence = detect_and_predict(img_rgb)

            with self.lock:
                self.last_emotion = emotion
                self.last_confidence = confidence

            out_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    result_placeholder = st.empty()

    if webrtc_ctx.state.playing:
        last_shown = None
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.lock:
                    emotion = webrtc_ctx.video_processor.last_emotion
                    confidence = webrtc_ctx.video_processor.last_confidence
                if emotion:
                    result_placeholder.markdown(
                        f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>",
                        unsafe_allow_html=True
                    )
                    if emotion != last_shown:
                        st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                        if len(st.session_state.history) > 5:
                            st.session_state.history.pop(0)
                        last_shown = emotion
                else:
                    result_placeholder.markdown(
                        "<div class='alert-box'>⚠️ No face detected. Center your face in the frame.</div>",
                        unsafe_allow_html=True
                    )
            time.sleep(0.5)

    st.caption(
        "Note: on some networks (corporate firewalls / strict NAT) the live stream may fail to "
        "connect with only a STUN server — a TURN server would be needed for full reliability."
    )

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
