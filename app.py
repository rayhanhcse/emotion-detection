import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# -----------------------------
# মডেল লোড
# -----------------------------
model = load_model("emotion.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Face Emotion Detection (Upload Image Only)")
st.write("Upload an image and detect emotions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    img_resized = image.resize((48,48))
    img_array = np.array(img_resized).reshape(1,48,48,1)/255.0

    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    st.image(image, caption=f"Predicted Emotion: {emotion}", use_column_width=True)
else:
    st.write("Upload an image to detect emotions")
