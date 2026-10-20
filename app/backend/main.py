"""
Kiosk Guard backend.

Loads one or two exported ONNX PAD models and exposes /verify, which takes a
base64 face crop and returns live/spoof + confidence for each model, so the demo
can show the baseline and the hardened model side by side on the same image.

Put your exported files next to this file:
    model_hardened.onnx   (required)
    model_baseline.onnx   (optional, enables the side-by-side comparison)

Run:
    cd app/backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""
import base64
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

HERE = Path(__file__).parent
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_FILES = {
    "hardened": ["model_hardened.onnx", "model.onnx"],
    "baseline": ["model_baseline.onnx"],
}
sessions = {}
for name, candidates in MODEL_FILES.items():
    for fname in candidates:
        if (HERE / fname).exists():
            sessions[name] = ort.InferenceSession(str(HERE / fname), providers=["CPUExecutionProvider"])
            break
if not sessions:
    raise RuntimeError("No ONNX model found. Put model_hardened.onnx (and model_baseline.onnx) in app/backend/")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VerifyRequest(BaseModel):
    image_b64: str  # base64 JPEG/PNG of an already-cropped face (data URL prefix is fine)


def preprocess(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def score(session, x):
    logit = float(session.run(["logit"], {"input": x})[0].reshape(-1)[0])
    prob_live = 1.0 / (1.0 + np.exp(-logit))
    return {
        "is_live": bool(prob_live > 0.5),
        "confidence": float(prob_live if prob_live > 0.5 else 1 - prob_live),
        "prob_live": float(prob_live),
    }


@app.post("/verify")
def verify(req: VerifyRequest):
    x = preprocess(req.image_b64)
    results = {name: score(sess, x) for name, sess in sessions.items()}
    primary = results.get("hardened") or next(iter(results.values()))
    return {**primary, "models": results}


@app.get("/health")
def health():
    return {"status": "ok", "models": list(sessions)}
