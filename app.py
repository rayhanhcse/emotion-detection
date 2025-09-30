import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import urllib.request
import time

st.set_page_config(page_title="RH-FED", layout="centered")

# -----------------------------
# Load Haar Cascade
# -----------------------------
cascade_file = "haarcascade_frontalface_default.xml"
if not st.session_state.get("cascade_loaded", False):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)
    st.session_state["cascade_loaded"] = True

face_cascade = cv2.CascadeClassifier(cascade_file)

# -----------------------------
# Load Model
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = load_model("emotion_new.h5", compile=False)

emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# App UI
# -----------------------------
st.title("Auto Real-time Face Emotion Detection")
st.write("Webcam mode: Detect face emotions automatically in real-time.")

start = st.checkbox("Start Webcam Detection")

FRAME_WINDOW = st.image([])

while start:
    # Capture frame automatically
    uploaded_image = st.camera_input("Webcam active - detecting faces automatically")
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        frame = np.array(image)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.reshape(1,48,48,1)/255.0
            prediction = st.session_state.model.predict(roi)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        FRAME_WINDOW.image(frame, channels="RGB")
    
    # Small delay for smoother updates
    time.sleep(0.2)
