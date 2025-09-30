import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import urllib.request
import time

st.set_page_config(page_title="রিয়েল-টাইম ফেস ইমোশন ডিটেকশন", layout="centered")

# -----------------------------
# Haar Cascade লোড
# -----------------------------
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_file)

face_cascade = cv2.CascadeClassifier(cascade_file)

# -----------------------------
# Keras মডেল লোড
# -----------------------------
model = load_model("emotion_new.h5", compile=False)
emotion_labels = ['রাগ', 'খুশি', 'নিরপেক্ষ', 'দুঃখ', 'অবাক']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("রিয়েল-টাইম ফেস ইমোশন ডিটেকশন")
st.write("ওয়েবক্যাম ব্যবহার করে মুখের অভিব্যক্তি সনাক্ত করুন।")

# -----------------------------
# ওয়েবক্যাম মোড
# -----------------------------
FRAME_WINDOW = st.image([])
run = st.checkbox('ওয়েবক্যাম চালু করুন')

while run:
    # Streamlit ক্যামেরা থেকে ছবি নিন
    uploaded_image = st.camera_input("আপনার মুখ ক্যাপচার করুন")
    if uploaded_image is not None:
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

        # ছবিটি আপডেট করুন
        FRAME_WINDOW.image(frame, channels="RGB")

    # হালকা বিলম্ব → smoother আপডেট
    time.sleep(0.1)
