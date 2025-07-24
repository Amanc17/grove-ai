import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastai.learner import load_learner
from fastai.vision.all import PILImage
import uvicorn

# ---------------------------
# 1. Download model if needed
EXPORT_PATH = "export.pkl"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1pyNscQnv5GPQibIJHvAgiPn-BXSh_cO4"

def download_model():
    if not os.path.exists(EXPORT_PATH):
        print("Downloading model from Google Drive...")
        r = requests.get(DRIVE_URL)
        with open(EXPORT_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

download_model()

# ---------------------------
# 2. Load model
try:
    learner = load_learner(EXPORT_PATH)
except Exception as e:
    print("Error loading model:", e)
    learner = None

# ---------------------------
# 3. FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if learner is None:
        return {"error": "Model not loaded"}
    try:
        img = PILImage.create(await file.read())
        pred, idx, probs = learner.predict(img)
        return {
            "disease": str(pred),
            "confidence": float(probs[idx]),
            "description": f"Predicted class: {pred} with {probs[idx]:.2%} confidence."
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "ok"}

# Optional: For local dev only
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
