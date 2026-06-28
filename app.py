import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

st.set_page_config(page_title="AnimalFound AI", page_icon="🐾")

st.title("🐶🐱 AnimalFound APP BY BUNNY")

st.write("Upload an image and AI will predict the object")

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet labels (simple version)
import json
import urllib.request

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(LABELS_URL).read().decode().splitlines()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    st.write("🤖 AI analyzing...")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top5 = torch.topk(probabilities, 3)

    st.success("Top Predictions:")

    for i in range(3):
        idx = top5.indices[i].item()
        score = top5.values[i].item()
        st.write(f"{labels[idx]} - {round(score*100, 2)}%")
