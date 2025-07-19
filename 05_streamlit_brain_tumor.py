import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI scan (grayscale image) to predict if a tumor is present.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 1) / 255.0
    return img_array

# ✅ Load full model
model = tf.keras.models.load_model("brain_tumor_model.keras")

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
