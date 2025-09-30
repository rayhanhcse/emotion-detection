import streamlit as st
#import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# -----------------------------
# মডেল লোড
# -----------------------------
model = load_model("emotion.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_labels =  ['Angry','Happy','Neutral','Sad','Surprise']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Face Emotion Detection Web App By Rayhan Hussain")
st.write("Use Webcam or Upload Image to detect emotions")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Mode", ["Webcam", "Upload Image"])

# -----------------------------
# Webcam Mode
# -----------------------------
if mode == "Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.reshape(1,48,48,1)/255.0
            prediction = model.predict(roi)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# -----------------------------
# Upload Image Mode
# -----------------------------
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image.convert('RGB'))

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

