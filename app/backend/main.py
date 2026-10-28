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

# Decision thresholds on P(live), chosen on VALIDATION (the operating points in the report).
# thresholds.json next to this file: {"baseline": 0.35, "hardened": 0.75}; missing -> 0.5.
THRESHOLDS = {"baseline": 0.5, "hardened": 0.5}
if (HERE / "thresholds.json").exists():
    import json
    THRESHOLDS.update(json.loads((HERE / "thresholds.json").read_text()))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VerifyRequest(BaseModel):
    image_b64: str  # base64 JPEG/PNG of an already-cropped face (data URL prefix is fine)
    condition: str = ""   # optional, for the real-world test log: e.g. "phone replay, max brightness"
    truth: str = ""       # optional: "live" or "spoof" (what was really in front of the camera)


def preprocess(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def score(session, x, threshold=0.5):
    logit = float(session.run(["logit"], {"input": x})[0].reshape(-1)[0])
    prob_live = 1.0 / (1.0 + np.exp(-logit))
    return {
        "is_live": bool(prob_live > threshold),
        "confidence": float(prob_live if prob_live > threshold else 1 - prob_live),
        "prob_live": float(prob_live),
        "threshold": float(threshold),
    }


@app.post("/verify")
def verify(req: VerifyRequest):
    x = preprocess(req.image_b64)
    results = {name: score(sess, x, THRESHOLDS.get(name, 0.5)) for name, sess in sessions.items()}
    primary = results.get("hardened") or next(iter(results.values()))
    if req.condition and req.truth:
        log_trial(req.condition, req.truth, results)
    return {**primary, "models": results}


LOG = HERE / "test_log.csv"


def log_trial(condition, truth, results):
    """Append one real-world test trial to test_log.csv (one row per verify)."""
    import csv
    import datetime
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "condition", "truth", "baseline_p_live", "baseline_says", "hardened_p_live", "hardened_says"])
        b, h = results.get("baseline", {}), results.get("hardened", {})
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), condition, truth,
                    round(b.get("prob_live", float("nan")), 4), "live" if b.get("is_live") else "spoof",
                    round(h.get("prob_live", float("nan")), 4), "live" if h.get("is_live") else "spoof"])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(sessions), "thresholds": THRESHOLDS}
