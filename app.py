import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="AnimalFound AI", page_icon="🐾")

st.title("🐶🐱 AnimalFound Real AI App")

st.write("Upload an image and AI will predict what it sees")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer hf_demo"}

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🤖 AI is analyzing...")

    img_bytes = uploaded_file.getvalue()
    result = query(img_bytes)

    if isinstance(result, list):
        st.success("Top Predictions:")

        for item in result[:3]:
            st.write(f"{item['label']} - {round(item['score']*100, 2)}%")

    else:
        st.error("Model loading... please wait a few seconds and retry")
