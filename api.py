import os
from tempfile import NamedTemporaryFile

import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from fastai.learner import load_learner
from fastai.vision.core import PILImage

MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?id=1pyNscQnv5GPQibIJHvAgiPn-BXSh_cO4")
MODEL_PATH = os.getenv("MODEL_PATH", "export.pkl")


ALLOWED_ORIGINS = [
    "https://groveai.org",
    "https://www.groveai.org",
    "https://groveai.vercel.app",
]

if os.getenv("ALLOW_ALL_ORIGINS", "0") == "1":
    ALLOWED_ORIGINS = ["*"]

app = FastAPI(title="GroveAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

learner = None  # will be populated on startup


def download_model():
    """Download model from Drive if it's not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("[startup] Model not found; downloading from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("[startup] Download complete.")


def load_model():
    """Load the FastAI learner (forced to CPU)."""
    global learner
    print("[startup] Loading learner...")
    # cpu=True forces map_location='cpu' under the hood
    learner = load_learner(MODEL_PATH, cpu=True)
    print("[startup] Learner ready.")


@app.on_event("startup")
def on_startup():
    
    try:
        download_model()
        load_model()
    except Exception as e:
        # If startup fails, log it loudly so Render/Vercel logs show it
        print(f"[startup] Failed to initialize model: {e}")
        


@app.get("/")
def root():
    return {"message": "Plant Disease Detection API is running!"}


@app.get("/healthz")
def healthz():
    """Lightweight health check endpoint."""
    return {"status": "ok", "model_loaded": bool(learner)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic validation
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload JPG, PNG, or WebP.")

    if learner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few seconds.")

    tmp_path = None
    try:
        # Save to a temp file (fastai expects a path-like)
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Try loading via fastai+PIL
        try:
            img = PILImage.create(tmp_path)
        except Exception as e:
            # Common cause: HEIC/HEIF uploaded and Pillow can't decode it
            # Option: install pillow-heif and convert — for now, return a clear error.
            raise HTTPException(
                status_code=400,
                detail="Could not read the image. If this is an HEIC photo, please convert it to JPG/PNG and try again.",
            ) from e

        pred, idx, probs = learner.predict(img)

        # Return normalized response (confidence 0..1)
        return {
            "disease": str(pred),
            "confidence": float(probs[idx]),
            "description": f"Predicted {pred} with confidence {probs[idx]:.2f}",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Any other server error
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
