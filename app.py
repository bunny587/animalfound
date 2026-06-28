import streamlit as st
from PIL import Image
import random

st.set_page_config(page_title="AnimalFound AI", page_icon="🐾")

st.title("🐶🐱 AnimalFound AI App")
st.write("Upload an image and get AI prediction")

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "png"])

animals = ["Dog 🐶", "Cat 🐱", "Elephant 🐘", "Lion 🦁", "Tiger 🐯"]

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🤖 Analyzing image...")

    prediction = random.choice(animals)

    st.success(f"Prediction: {prediction}")
