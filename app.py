import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="AnimalFound AI", page_icon="🐾")

st.title("🐶🐱 AnimalFound REAL AI Model")

st.write("Upload an image and AI will predict what it sees")

# Load pretrained model (runs inside app)
model = tf.keras.applications.MobileNetV2(weights="imagenet")
decode = tf.keras.applications.mobilenet_v2.decode_predictions

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    st.write("🤖 AI is thinking...")

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    preds = model.predict(img)
    results = decode(preds, top=3)[0]

    st.success("Top Predictions:")

    for _, name, score in results:
        st.write(f"{name} - {round(score*100, 2)}%")
