import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="AnimalFound AI", page_icon="🐾")

st.title("🐶🐱 AnimalFound Real AI Model")

st.write("Upload an image and AI will predict what it is")

# Load pretrained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# labels helper
decode = tf.keras.applications.mobilenet_v2.decode_predictions

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🤖 Predicting...")

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    results = decode(predictions, top=3)[0]

    st.success("Top Predictions:")

    for i, (id, name, score) in enumerate(results):
        st.write(f"{i+1}. {name} - {round(score*100, 2)}%")
