"""Shared TFLite Cat/Dog/Human predictor for Vercel Python functions."""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "model", "cat_dog_model.tflite")
CLASS_NAMES_PATH = os.path.join(HERE, "..", "model", "class_names.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

FUN_FACTS = {
    "cat": "Cats use their whiskers to sense nearby objects.",
    "dog": "Dogs have an excellent sense of smell.",
    "human": "Humans have around 100 billion brain cells.",
}

_interpreter = None


def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            return [str(x).lower() for x in json.load(f)]
    except Exception:
        return ["cat", "dog", "human"]


def get_interpreter():
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predict_image(img, class_names):
    interp = get_interpreter()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]["index"], preprocess(img))
    interp.invoke()
    out = interp.get_tensor(output_details[0]["index"])[0]
    probs = {name: float(prob) for name, prob in zip(class_names, out)}
    idx = int(np.argmax(out))
    confidence = float(out[idx])
    prediction = class_names[idx]
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "sure": confidence >= CONFIDENCE_THRESHOLD,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "fun_fact": FUN_FACTS.get(prediction, ""),
    }


def decode_image(data):
    if data.startswith("data:"):
        data = data.split(",", 1)[1] if "," in data else data
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw))