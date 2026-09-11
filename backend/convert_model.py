"""Convert cat_dog_model.keras to a lightweight TFLite model for hosting."""

import os

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "cat-dog-camera", "model", "cat_dog_model.keras")
OUT_DIR = os.path.join(HERE, "model")
OUT = os.path.join(OUT_DIR, "cat_dog_model.tflite")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading keras model from:", os.path.realpath(SRC))
model = tf.keras.models.load_model(SRC)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open(OUT, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite model to:", os.path.realpath(OUT))
print("Size:", round(os.path.getsize(OUT) / (1024 * 1024), 1), "MB")