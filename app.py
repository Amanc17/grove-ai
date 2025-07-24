# app.py
import streamlit as st
from fastai.vision.all import *
import PIL

# 1. Load the Model (cache for performance)
@st.cache_resource
def load_learner():
    try:
        learn = load_learner("plant-resnet18-fastai.pth")  # .pth extension for FastAI >=2.7
        return learn
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# 2. Streamlit UI
st.set_page_config(page_title="Grove - Plant Disease Detector", layout="centered")
st.title("Grove: Plant Disease Classifier")
st.write("Upload a plant leaf image below and see what disease (if any) the AI finds!")

learner = load_learner()
if learner is None:
    st.stop()

# 3. Image Upload and Prediction
uploaded = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
if uploaded:
    img = PIL.Image.open(uploaded)
    st.image(img, caption="Your uploaded image", use_column_width=True)
    with st.spinner("Analyzing..."):
        pred, pred_idx, probs = learner.predict(img)
        st.success(f"**Prediction:** {pred}")
        st.info(f"**Confidence:** {probs[pred_idx]:.2%}")