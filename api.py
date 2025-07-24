# save as api.py
from fastai.vision.all import load_learner, PILImage
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

model = load_learner("export.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = PILImage.create(await file.read())
    pred, pred_idx, probs = model.predict(img)
    return {
        "prediction": str(pred),
        "confidence": float(probs[pred_idx])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
