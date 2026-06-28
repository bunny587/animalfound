import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pretrained model (MobileNet - already trained on animals & objects)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

st.title("🐶🐱 Animal Image Predictor AI")

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image for model
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    st.write("🔍 Prediction:")
    st.success(decoded[0][0][1])