import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import time
import threading
from collections import deque
import pandas as pd
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
body {
    background: linear-gradient(270deg, #f8f9fa, #ecf0f1, #dfe6e9);
    background-size: 600% 600%;
    animation: gradientBG 12s ease infinite;
    font-family: 'Segoe UI', sans-serif;
}
@keyframes gradientBG {0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.navbar {
    width: 100%; padding: 20px; text-align: center; font-size: 26px; font-weight: bold;
    color: white; border-radius: 0 0 20px 20px;
    background: linear-gradient(-45deg, #1abc9c, #3498db, #9b59b6, #e74c3c);
    background-size: 400% 400%; animation: gradientAnimation 15s ease infinite;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35); text-shadow: 1px 2px 6px rgba(0,0,0,0.3);
}
@keyframes gradientAnimation {0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.title {text-align:center;font-size:44px;font-weight:bold;color:#2c3e50;margin-top:25px;text-shadow:2px 3px 8px rgba(0,0,0,0.15);}
.subtitle {text-align:center;font-size:20px;color:#555;margin-bottom:35px;}
.result-box {
    margin-top: 20px; padding: 20px; border-radius: 15px; background: rgba(52,152,219,0.2);
    border:1px solid #3498db; color:#2c3e50; font-weight:bold; text-align:center; font-size:22px;
    box-shadow: 0px 8px 20px rgba(52,152,219,0.4); backdrop-filter: blur(6px);
}
.alert-box {
    background: linear-gradient(45deg,#f39c12,#e67e22); color:white; font-weight:bold;
    padding:15px; border-radius:15px; text-align:center; margin-top:20px; font-size:18px;
    box-shadow: 0px 6px 18px rgba(243,156,18,0.5);
}
.footer {
    position: fixed; left: 0; bottom: 0; width: 100%; text-align: center; color:#333;
    font-size:14px; padding:14px; background: rgba(255,255,255,0.7);
    border-top:1px solid rgba(200,200,200,0.3); backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
st.markdown("<div class='navbar'>Welcome to Emotion World!</div>", unsafe_allow_html=True)

# ------------------------------
# Sidebar
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
# Cached, defensive resource loading
# Using st.cache_resource means these load ONCE per app instance instead of on
# every rerun, which matters a lot for RAM/CPU on Streamlit Cloud's free tier.
@st.cache_resource(show_spinner="Loading face detector...")
def get_face_cascade():
    cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        raise RuntimeError("Haar cascade failed to load from OpenCV's bundled data.")
    return cascade

@st.cache_resource(show_spinner="Loading emotion model (this can take a moment)...")
def get_model():
    return load_model("emotion_new.h5", compile=False)

try:
    face_cascade = get_face_cascade()
    model = get_model()
except Exception as e:
    st.error(
        "⚠️ The app failed to load its face detector or emotion model.\n\n"
        f"Details: `{e}`\n\n"
        "Common causes: the model file is missing/corrupted in the repo, "
        "or the app ran out of memory while loading it. Check 'Manage app' logs for the full traceback."
    )
    st.stop()

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
st.markdown("<div class='title'>AI-Based Facial Emotion Analysis System Using Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered | By Rayhan Hussain</div>", unsafe_allow_html=True)

# ------------------------------
# Mode Buttons
if "mode" not in st.session_state:
    st.session_state.mode = None

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("👉 **Snapshot from your webcam**")
    if st.button("📸 Webcam Snapshot"):
        st.session_state.mode = "Webcam"
with col2:
    st.markdown("👉 **Upload an image**")
    if st.button("📂 Upload Image Detection"):
        st.session_state.mode = "Upload Image"
with col3:
    st.markdown("👉 **Live detection with charts**")
    if st.button("🎥 Live Emotion Analysis"):
        st.session_state.mode = "Live"

# ------------------------------
# Detect & Predict (expects RGB frame)
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

        for i in range(6, 0, -2):
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (52, 152, 219), i+4)
            alpha = 0.25 * (i/6)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (52, 152, 219), 2)
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 4)
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (231,76,60), 2)
    return frame, detected_emotion, confidence

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Webcam Snapshot Mode
if st.session_state.mode == "Webcam":
    st.info("📸 Turn on your webcam and capture your face")
    uploaded_image = st.camera_input("Click below to capture your face 👇")
    if uploaded_image:
        with st.spinner("🔍 Analyzing emotions..."):
            image = Image.open(uploaded_image)
            frame = np.array(image.convert('RGB'))
            processed, emotion, confidence = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history) > 5: st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

# ------------------------------
# Upload Mode
elif st.session_state.mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload your image (jpg,jpeg,png)", type=["jpg","jpeg","png"])
    if uploaded_file:
        with st.spinner("🔍 Detecting emotions..."):
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            processed, emotion, confidence = detect_and_predict(frame)
            st.image(processed, channels="RGB", use_container_width=True)
            if emotion:
                st.markdown(f"<div class='result-box'>{emoji_map[emotion]} Detected Emotion: <b>{emotion}</b> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
                st.info(quotes[emotion])
                st.session_state.history.append(f"{emoji_map[emotion]} {emotion}")
                if len(st.session_state.history) > 5: st.session_state.history.pop(0)
            else:
                st.markdown("<div class='alert-box'>⚠️ No face detected. Please try again with a clear image or better lighting!</div>", unsafe_allow_html=True)

# ------------------------------
# Live Mode: real-time video + live bar/line charts
elif st.session_state.mode == "Live":
    st.info("🎥 Allow camera access for continuous live emotion detection with real-time analytics.")

    skip_n = st.slider(
        "Process every Nth frame (higher = lighter on CPU/RAM, less smooth labels)",
        min_value=1, max_value=10, value=4
    )

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.last_emotion = None
            self.last_confidence = None
            self.frame_count = 0
            self.skip_n = skip_n
            # rolling log of (timestamp, emotion, confidence) for charts
            self.log = deque(maxlen=200)

        def recv(self, frame):
            self.frame_count += 1
            img_bgr = frame.to_ndarray(format="bgr24")

            # Only run the (expensive) model every Nth frame to save CPU/RAM;
            # other frames pass through with the last known box/label untouched.
            if self.frame_count % self.skip_n == 0:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                processed_rgb, emotion, confidence = detect_and_predict(img_rgb)
                out_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.last_emotion = emotion
                    self.last_confidence = confidence
                    if emotion:
                        self.log.append((time.time(), emotion, confidence))
            else:
                out_bgr = img_bgr

            return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    result_placeholder = st.empty()
    st.subheader("📊 Live Analytics")
    chart_col1, chart_col2 = st.columns(2)
    bar_placeholder = chart_col1.empty()
    line_placeholder = chart_col2.empty()

    if webrtc_ctx.state.playing:
        last_shown = None
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.lock:
                    emotion = webrtc_ctx.video_processor.last_emotion
                    confidence = webrtc_ctx.video_processor.last_confidence
                    log_snapshot = list(webrtc_ctx.video_processor.log)

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

                if log_snapshot:
                    df = pd.DataFrame(log_snapshot, columns=["time", "emotion", "confidence"])

                    # Bar chart: how often each emotion has appeared this session
                    counts = df["emotion"].value_counts().reindex(emotion_labels, fill_value=0)
                    bar_placeholder.bar_chart(counts)

                    # Line chart: confidence trend over the last readings
                    recent = df.tail(50).copy()
                    recent["reading"] = range(len(recent))
                    line_placeholder.line_chart(recent.set_index("reading")["confidence"])

            time.sleep(0.5)

    st.caption(
        "Note: on strict corporate/school networks the live stream may fail to connect with only "
        "a STUN server — a TURN server would be needed for full reliability. If the app feels slow "
        "or crashes, raise the 'process every Nth frame' slider to reduce load."
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
