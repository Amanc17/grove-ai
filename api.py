import os
import requests
from fastai.learner import load_learner
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

EXPORT_PATH = "export.pkl"
DRIVE_FILE_ID = "1pyNscQnv5GPQibIJHvAgiPn-BXSh_cO4"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename}.")

download_if_not_exists(EXPORT_PATH, DRIVE_URL)

# Load model (cache so it only loads once)
learner = load_learner(EXPORT_PATH)

# --- FastAPI setup ---
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        pred, pred_idx, probs = learner.predict(img)
        return JSONResponse({
            "prediction": str(pred),
            "confidence": float(probs[pred_idx])
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Plant Disease AI API is running."}
