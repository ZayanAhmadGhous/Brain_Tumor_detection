import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image


# Title
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI scan (grayscale image) to predict if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# Preprocess image
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 1) / 255.0
    return img_array


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])


# Load weights
model.load_weights("model.weights.h5")  # ✅ correct extension


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    label = "🧠 Tumor Detected" if class_index == 1 else "✅ No Tumor Detected"
    confidence = prediction[0][class_index] * 100

    st.subheader(label)
    st.write(f"Confidence: **{confidence:.2f}%**")
