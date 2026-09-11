"""
🐱🐶🧑 Cat / Dog / Human detection API.

Runs a lightweight TFLite MobileNetV2 classifier on hosted servers (Render).

Endpoints:
    GET  /            -> API info
    GET  /health      -> server + model status
    POST /predict     -> { image: "<base64 jpeg/png>" } -> classification
"""

import base64
import io
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # local fallback

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "cat_dog_model.tflite")
CLASS_NAMES_PATH = os.path.join(HERE, "model", "class_names.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

FUN_FACTS = {
    "cat": "Cats use their whiskers to sense nearby objects.",
    "dog": "Dogs have an excellent sense of smell.",
    "human": "Humans have around 100 billion brain cells.",
}

app = FastAPI(title="Cat / Dog / Human Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64-encoded image (optionally a data URL)


def _load_class_names():
    try:
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            return [str(x).lower() for x in json.load(f)]
    except Exception:
        return ["cat", "dog", "human"]


CLASS_NAMES = _load_class_names()

_interpreter = None


def get_interpreter():
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    arr = (arr / 127.5) - 1.0  # MobileNetV2 preprocess_input
    return np.expand_dims(arr, axis=0)


def predict_image(img):
    interp = get_interpreter()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    tensor = preprocess(img)
    interp.set_tensor(input_details[0]["index"], tensor)
    interp.invoke()
    out = interp.get_tensor(output_details[0]["index"])[0]

    probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, out)}
    idx = int(np.argmax(out))
    confidence = float(out[idx])
    prediction = CLASS_NAMES[idx]
    sure = confidence >= CONFIDENCE_THRESHOLD

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "sure": sure,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "fun_fact": FUN_FACTS.get(prediction, ""),
    }


@app.get("/")
def root():
    return {
        "name": "Cat / Dog / Human Detector API",
        "classes": CLASS_NAMES,
        "endpoints": ["/predict", "/health"],
    }


@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH)
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model": os.path.basename(MODEL_PATH),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    data = req.image
    if data.startswith("data:"):
        data = data.split(",", 1)[1] if "," in data else data
    try:
        raw = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")
    try:
        return predict_image(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")