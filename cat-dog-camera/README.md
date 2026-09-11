# 🐱🐶 AI LIVE CAT VS DOG DETECTOR

A real-time camera AI project. You point the computer's webcam at a real cat or dog, and the AI shows **CAT** or **DOG** live on the camera screen, together with a confidence score.

Built for a Class 4/5 school presentation. It runs 100% locally on your computer — no internet, no paid services, no cloud AI.

---

## 1. What the project does

- Opens the computer's webcam.
- Mirrors the camera image (like a selfie mirror).
- Reads the live video frame by frame.
- Runs a trained AI image classifier on each frame to decide: **CAT** or **DOG**.
- Shows the prediction, confidence, and a fun fact on the screen.
- Uses simple smoothing so the prediction does not jump around wildly.
- If the AI is not confident, it honestly says **🤔 NOT SURE**.

Does NOT use file names, random choices, or hardcoded answers. Every prediction comes from the trained AI model.

---

## 2. How the webcam works

The program uses OpenCV:

```python
cv2.VideoCapture(0)
```

- `0` means the default webcam.
- The image is captured in **BGR** colour order (the way OpenCV reads video).
- It is flipped horizontally with `cv2.flip(frame, 1)` to work like a mirror.
- The resolution is set to **640 × 480** for smooth performance.
- The model only runs every **3rd frame**, so the camera stays smooth.

---

## 3. How AI classification works

For every processed video frame the program:

1. Captures the frame.
2. Converts BGR → RGB.
3. Resizes the image to **224 × 224** (the size MobileNetV2 expects).
4. Normalises the image with MobileNetV2's own preprocessor (`preprocess_input`).
5. Runs the trained model.
6. The model outputs two probabilities: Cat and Dog.
7. The higher probability wins.
8. If that probability is at least **70%**, the prediction is shown.

```
This is a REAL learned model, not a guess.
```

---

## 4. Dataset structure

```
cat-dog-camera/
│
├── main.py
├── train_model.py
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── cat/        <- put CAT photos here
│   └── dog/        <- put DOG photos here
│
└── model/
    ├── cat_dog_model.keras
    └── class_names.json
```

---

## 5. How to prepare cat images

1. Make a folder: `dataset/cat`
2. Add many photos of cats: facing the camera, from different angles, at different distances, and in different rooms (lighting matters!).
3. Use **at least 50 photos** of cats for a better model (more is better).
4. Get them from your own camera, downloaded free stock photos, or pictures you took yourself.

## 6. How to prepare dog images

Do the same for dogs:

1. Make a folder: `dataset/dog`
2. Add many photos of dogs (different breeds, angles, lighting).
3. Use **at least 50 photos** of dogs.

The model learns better when both classes have a similar number of photos.

---

## 7. How to train the model

Install the required packages:

```
pip install -r requirements.txt
```

Then run:

```
python train_model.py
```

What happens:

1. Loads all cat and dog pictures.
2. Resizes them to 224 × 224.
3. Normalises them for MobileNetV2.
4. Adds data augmentation (flip, rotate, zoom, brightness) so the model does not overfit.
5. Splits the data into 80% training and 20% validation.
6. Loads **MobileNetV2** with pre-trained ImageNet weights.
7. Freezes the base model and adds a new classifier head that outputs exactly 2 classes: CAT and DOG.
8. Trains the new head.
9. Fine-tunes the top part of MobileNetV2 at a lower learning rate.
10. Prints the **actual** training and validation accuracy (no fake numbers).
11. Saves:

```
model/cat_dog_model.keras
model/class_names.json
```

---

## 8. How to run the camera detector

After training, run:

```
python main.py
```

The webcam opens and the program shows:

- A title bar: `🐱🐶 AI CAT VS DOG DETECTOR`
- A prediction panel with `🤖 AI PREDICTION`
- `🐱 CAT DETECTED!` or `🐶 DOG DETECTED!`
- `Confidence: 94%`
- A fun fact
- `🤔 NOT SURE` + `Confidence: 54%` when the model cannot decide

---

## 9. Keyboard controls

| Key | Action |
|-----|--------|
| `Q` | Quit the program |
| `R` | Reset prediction history (clears the AI's memory, shows "Waiting for animal...") |
| `S` | Save the current camera frame to `captured_result.jpg` |

Screenshots are **only** saved when you press `S`.

---

## 10. Confidence score

The model gives two probabilities for each frame, e.g. `Cat 0.94 / Dog 0.06`.

- `Confidence >= 70%` → show the prediction (CAT or DOG).
- `Confidence < 70%` → show `🤔 NOT SURE` and a message: *"Please show a clear cat or dog."*

The program also smooths predictions: it keeps the last 10 results and only updates the screen when at least 6 of them agree. This prevents the output from flickering CAT-dog-CAT-dog.

---

## 11. Limits (important, be honest!)

This is an educational project. The AI is **not guaranteed** to identify every cat or dog correctly. Accuracy depends on:

- Training dataset size and variety
- Lighting
- Camera quality
- Animal position and distance
- Background
- Breed of the animal

A Cat/Dog binary classifier can also wrongly classify unrelated objects. That is why it says **NOT SURE** when the confidence is low — reliability matters more than always giving an answer.

---

## 12. Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install -r requirements.txt` fails | Make sure Python 3.9–3.12 is installed and try again. |
| "Trained AI model not found" | You must run `python train_model.py` first. |
| "Could not open webcam" | Check camera connection, permissions, and that no other app (Zoom/Meet/browser) is using the camera. |
| Prediction flickers | Give the smoothing time (about 1 second) — it averages the last 10 frames. |
| It says NOT SURE a lot | Add more varied training photos and re-train. |
| Captured picture looks like a selfie | That is correct, the camera is mirrored on purpose. |

---

## 13. Future improvements

Possible ideas:

- Add more classes (e.g. bird, fish, rabbit).
- Detect the animal region with a segmentation model and classify only that region.
- Show a live confidence graph / history chart.
- Play a sound when a cat or dog is detected.
- Make it run on a Raspberry Pi with a USB camera.

---

## Class 4/5 presentation — simple explanation

> "My project uses a webcam to see an animal.
>
> I trained an AI model using many pictures of cats and dogs.
>
> The model learned visual patterns from those examples.
>
> When I show an animal to the camera, the computer takes the camera image and gives probabilities for cat and dog.
>
> The result with the higher confidence becomes the AI prediction.
>
> If the AI is not confident, it says 'I'm not sure'."

---

## Commands summary

```
pip install -r requirements.txt

python train_model.py

python main.py
```