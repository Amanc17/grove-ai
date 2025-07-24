import os
import gdown
from fastapi import FastAPI, File, UploadFile
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from tempfile import NamedTemporaryFile
from starlette.responses import JSONResponse

MODEL_URL = "https://drive.google.com/uc?id=1pyNscQnv5GPQibIJHvAgiPn-BXSh_cO4"
MODEL_PATH = "export.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise

# Download and load model at startup
download_model()
learner = load_learner(MODEL_PATH)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        img = PILImage.create(tmp_path)
        pred, idx, probs = learner.predict(img)
        os.remove(tmp_path)

        return {
            "disease": str(pred),
            "confidence": float(probs[idx]),
            "description": f"Predicted {pred} with confidence {probs[idx]:.2f}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
