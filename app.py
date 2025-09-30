import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import urllib.request
import time

st.set_page_config(page_title="Real-time Face Emotion Detection By Rayhan Hussain", layout="centered")

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
st.title("Real-time Face Emotion Detection")
st.write("Webcam mode: Detect emotions in real-time!")

# -----------------------------
# Webcam Real-time Prediction
# -----------------------------
FRAME_WINDOW = st.image([])  # Placeholder for webcam frames
run = st.checkbox('Start Webcam')

while run:
    # Capture image from Streamlit webcam
    uploaded_image = st.camera_input("Capture your face")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        frame = np.array(image)

        # Convert to grayscale for face detection
        gray = cv2.c
