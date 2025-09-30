import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import urllib.request

st.set_page_config(page_title="Face Emotion Detection", layout="centered")

# -----------------------------
# Load Haar Cascade
# -----------------------------
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)

face_cascade = cv2.CascadeClassifier(cascade_file)

# -----------------------------
# Load Keras Model
# -----------------------------
model = load_model("emotion_new.h5", compile=False)
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Face Emotion Detection Web App")
st.write("Select a mode to detect emotions: Webcam or Upload Image.")

# -----------------------------
# Two buttons for mode selection
# -----------------------------
col1, col2 = st.columns(2)
use_webcam = col1.button("Use Webcam")
use_upload = col2.button("Upload Image")

# -----------------------------
# Webcam Mode
# -----------------------------
if use_webcam:
    FRAME_WINDOW = st.image([])
    uploaded_image = st.camera_input("Capture your face")
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        frame = np.array(image)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.reshape(1,48,48,1)/255.0
            prediction = model.predict(roi)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        FRAME_WINDOW.image(frame, channels="RGB")

# -----------------------------
# Upload Image Mode
# -----------------------------
elif use_upload:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.reshape(1,48,48,1)/255.0
            prediction = model.predict(roi)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        st.image(frame, channels="RGB")
