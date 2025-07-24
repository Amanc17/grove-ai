import streamlit as st
from fastai.vision.all import *
import urllib.request
import os
import pathlib

# --------- Download model file if not present ---------
MODEL_URL = "https://drive.google.com/file/d/1EXMSkYnjBlq1YtoCuc3TmmKtUz5Ko0f6/view?usp=sharing"  # Replace with your direct link
MODEL_PATH = "plant-resnet18-fastai.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded.")

# --------- Load the FastAI model ---------
@st.cache_resource
def load_learner():
    data_root = pathlib.Path.cwd()  # For deployment, keep things in current dir
    # You should specify your data structure or dummy dls as appropriate for your export
    dummy_dls = ImageDataLoaders.from_folder(
        path=data_root, train=".", valid=".", item_tfms=Resize(224), bs=1
    )
    learn = vision_learner(dummy_dls, resnet18)
    learn.load("plant-resnet18-fastai")  # Do NOT add .pth
    return learn

learner = load_learner()

# --------- Streamlit UI ---------
st.title("🌱 Grove: Plant Disease Classifier")
st.write("Upload a plant leaf image to identify diseases with AI.")

uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded:
    img = PILImage.create(uploaded)
    st.image(img, caption="Your uploaded image", use_column_width=True)
    with st.spinner("Analyzing..."):
        pred, idx, probs = learner.predict(img)
    st.success(f"**Prediction:** {pred}")
    st.info(f"**Confidence:** {probs[idx]:.4f}")
