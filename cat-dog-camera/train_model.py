"""
🐱🐶🧑 AI CAT VS DOG VS HUMAN — MODEL TRAINING

Trains a MobileNetV2 transfer-learning model to classify
CAT, DOG and HUMAN pictures.

Dataset folders used:
    dataset/cat/    -> cat pictures
    dataset/dog/    -> dog pictures
    dataset/human/  -> human pictures

Saved outputs:
    model/cat_dog_model.keras
    model/class_names.json

Run:
    python train_model.py
"""

import json
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 25
FINE_TUNE_TOP = 60
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42
MIN_IMAGES = 5
CLASSES = ["cat", "dog", "human"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTS))


def load_class_data(cls):
    folder = os.path.join(DATASET_DIR, cls)
    images = []
    names = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(folder, name)
        if cv2 is not None:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.utils.img_to_array(img).astype("uint8")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img.astype("float32"))
        names.append(name)
    return images, names


def load_all():
    xs, ys, _names = [], [], []
    for i, cls in enumerate(CLASSES):
        imgs, names = load_class_data(cls)
        xs.extend(imgs)
        ys.extend([i] * len(imgs))
    idx = np.random.RandomState(SEED).permutation(len(xs))
    xs = np.array(xs)[idx]
    ys = np.array(ys)[idx]
    n = len(xs)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    train_x = xs[:n_train]
    train_y = ys[:n_train]
    val_x = xs[n_train:n_train + n_val]
    val_y = ys[n_train:n_train + n_val]
    test_x = xs[n_train + n_val:]
    test_y = ys[n_train + n_val:]
    return train_x, train_y, val_x, val_y, test_x, test_y


def aug(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=0.25)
    x = tf.image.random_contrast(x, lower=0.7, upper=1.3)
    x = tf.image.random_saturation(x, lower=0.7, upper=1.3)
    x = tf.image.random_hue(x, max_delta=0.05)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return x


def make_ds(x, y, training):
    xp = preprocess_input(x)
    ds = tf.data.Dataset.from_tensor_slices((xp, y))
    if training:
        ds = ds.map(lambda img, lbl: (aug(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(len(CLASSES), activation="softmax", name="classifier")(x)
    model = models.Model(base_model.input, output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main():
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    counts = {cls: count_images(os.path.join(DATASET_DIR, cls)) for cls in CLASSES}

    print("================================================")
    print("🐱🐶🧑 AI CAT VS DOG VS HUMAN — MODEL TRAINING")
    print("================================================")
    print()
    for cls in CLASSES:
        print(f"{cls.capitalize():<6} pictures found : {counts[cls]}")
    print()

    if any(counts[cls] < MIN_IMAGES for cls in CLASSES):
        print("⚠️  Not enough pictures. Add photos to:")
        for cls in CLASSES:
            print(f"    {os.path.join(DATASET_DIR, cls)}")
        print("    Then run this script again:  python train_model.py")
        sys.exit(1)

    if any(counts[cls] < 50 for cls in CLASSES):
        print("💡  Tip: 50+ pictures per class usually give a more accurate model.")
        print()

    print("Loading and splitting images (train/val/test)...")
    train_x, train_y, val_x, val_y, test_x, test_y = load_all()
    print(f"  Train: {len(train_x)}   Val: {len(val_x)}   Test: {len(test_x)}")
    print()

    class_names = list(CLASSES)

    train_ds = make_ds(train_x, train_y, training=True)
    val_ds = make_ds(val_x, val_y, training=False)
    test_ds = make_ds(test_x, test_y, training=False)

    model, base_model = build_model()
    model.summary()

    early = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )

    print()
    print("Step 1: Training the new classifier layers...")
    print()
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early, reduce_lr],
        verbose=1,
    )

    print()
    print("Step 2: Fine-tuning part of the MobileNetV2 base...")
    print()
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_TOP]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early2 = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr2 = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )
    model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=[early2, reduce_lr2],
        verbose=1,
    )

    print()
    print("Evaluating the model (no augmentation)...")
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    pred = model.predict(test_ds, verbose=0)
    predy = np.argmax(pred, axis=1)
    print()
    for i, name in enumerate(class_names):
        mask = test_y == i
        if mask.sum():
            print(f"  Test {name} accuracy : {(predy[mask] == i).mean():.1%}  ({mask.sum()} imgs)")

    print()
    print("================================================")
    print("TRAINING RESULTS")
    print("================================================")
    print(f"Training accuracy   : {train_acc:.1%}")
    print(f"Validation accuracy : {val_acc:.1%}")
    print(f"Test accuracy       : {test_acc:.1%}")
    print()

    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("✅ Model saved to     :", MODEL_PATH)
    print("✅ Class names saved  :", CLASS_NAMES_PATH)
    print()
    print("Now run the live webcam detector:")
    print("    python main.py")


if __name__ == "__main__":
    main()
