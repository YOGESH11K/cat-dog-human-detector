"""
🐱🐶 AI CAT VS DOG DETECTOR

Live webcam detection of CAT vs DOG using a locally
trained MobileNetV2 model.

Run:
    python main.py
"""

import json
import os
import sys
import time
from collections import Counter, deque

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def missing(name):
    print("⚠️  Missing dependency:", name)
    print("    Install it with:  pip install -r requirements.txt")
    sys.exit(1)


try:
    import numpy as np
except ImportError:
    missing("NumPy")

try:
    import cv2
except ImportError:
    missing("OpenCV")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    missing("TensorFlow / Keras")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "cat_dog_model.keras")
CLASS_NAMES_PATH = os.path.join(HERE, "model", "class_names.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
WINDOW_SIZE = 10
MIN_MAJORITY = 6
PREDICT_EVERY = 3

TITLE = "🐱🐶 AI CAT VS DOG DETECTOR"

FUN_FACTS = {
    "cat": "Cats use their whiskers to sense nearby objects.",
    "dog": "Dogs have an excellent sense of smell.",
    "human": "Humans have around 100 billion brain cells.",
}

C_TITLE_BG = (26, 26, 80)
C_PANEL_BG = (22, 22, 66)
C_BORDER = (130, 130, 255)
C_DIVIDER = (90, 90, 220)
C_CONTROL_BG = (20, 20, 60)
C_TEXT = (255, 255, 255)
C_ACCENT = (110, 215, 255)
C_CAT = (0, 170, 255)
C_DOG = (255, 150, 40)
C_HUMAN = (110, 230, 130)
C_WARN = (0, 235, 255)
C_SAVED = (80, 220, 130)

EMOJI_FONT_PATHS = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "seguiemj.ttf"),
    r"C:\Windows\Fonts\seguiemj.ttf",
]
LATIN_FONT_PATHS = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"),
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

_FONT_CACHE = {}


def _font(kind, size):
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    paths = EMOJI_FONT_PATHS if kind == "emoji" else LATIN_FONT_PATHS
    font = None
    for path in paths:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None and kind == "latin":
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _is_emoji(ch):
    return ord(ch) > 0xFFFF


def _runs(text):
    result = []
    buf = ""
    cur = False

    def flush():
        nonlocal buf
        if buf:
            result.append((buf, cur))
            buf = ""

    for ch in text:
        e = _is_emoji(ch)
        if e != cur:
            flush()
            cur = e
        buf += ch
    flush()
    return result


class Overlay:
    def __init__(self, frame):
        self.frame = frame
        self.pil = None
        if PIL_OK:
            self.pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.draw = ImageDraw.Draw(self.pil)

    def _measure(self, text, size):
        total = 0.0
        for chunk, is_e in _runs(text):
            f = _font("emoji" if is_e else "latin", size)
            total += self.draw.textlength(chunk, font=f)
        return total

    def text(self, text, x, y, size, color, anchor="left"):
        rgb = (color[2], color[1], color[0])
        if self.pil is not None:
            if anchor == "center":
                ax = x - self._measure(text, size) / 2.0
            elif anchor == "right":
                ax = x - self._measure(text, size)
            else:
                ax = x
            for chunk, is_e in _runs(text):
                f = _font("emoji" if is_e else "latin", size)
                self.draw.text((ax, y), chunk, font=f, fill=rgb, anchor="la")
                ax += self.draw.textlength(chunk, font=f)
        else:
            clean = "".join(c for c in text if ord(c) < 128).strip()
            if not clean:
                return
            scale = max(0.35, size / 85.0)
            thick = max(1, size // 28)
            (w, h), _ = cv2.getTextSize(clean, cv2.FONT_HERSHEY_TRIPLEX, scale, thick)
            if anchor == "center":
                x0 = x - w // 2
            elif anchor == "right":
                x0 = x - w
            else:
                x0 = x
            cv2.putText(self.frame, clean, (int(x0), int(y + h)),
                        cv2.FONT_HERSHEY_TRIPLEX, scale, color, thick, cv2.LINE_AA)

    def wrap(self, text, max_width, size):
        if self.pil is not None:
            f = _font("latin", size)
            words = text.split(" ")
            lines = []
            cur = ""
            for word in words:
                trial = (cur + " " + word).strip()
                if self.draw.textlength(trial, font=f) <= max_width or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            return lines
        chars = max(10, int(max_width / (size * 0.50)))
        out = []
        while len(text) > chars:
            cut = text[:chars]
            sp = cut.rfind(" ")
            if sp > 0:
                cut, text = cut[:sp], text[sp + 1:]
            else:
                text = text[chars:]
            out.append(cut)
        out.append(text)
        return out

    def finish(self):
        if self.pil is not None:
            return cv2.cvtColor(np.array(self.pil), cv2.COLOR_RGB2BGR)
        return self.frame


def _rounded_filled(img, x1, y1, x2, y2, radius, color):
    r = max(radius, 1)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1)


def _rounded_outline(img, x1, y1, x2, y2, radius, color, thickness=2):
    r = max(radius, 1)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r + 1, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r + 1), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 180, 270, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 270, 360, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def predict(model, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")
    batch = preprocess_input(np.expand_dims(resized, axis=0))
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        print()
        print("⚠️  Trained AI model not found.")
        print()
        print("The file 'model/cat_dog_model.keras' does not exist.")
        print()
        print("To create your own AI model:")
        print("  1. Prepare pictures inside:")
        print("       dataset/cat/")
        print("       dataset/dog/")
        print("  2. Train the model:")
        print("       python train_model.py")
        print("  3. Then run this program:")
        print("       python main.py")
        print()
        sys.exit(1)
    try:
        return load_model(MODEL_PATH)
    except Exception as exc:
        print()
        print("⚠️  Could not load the trained AI model.")
        print("    Error:", exc)
        print("    Try training it again:  python train_model.py")
        print()
        sys.exit(1)


def load_class_labels():
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return [str(x).lower() for x in data.get("class_names", ["cat", "dog"])]
            return [str(x).lower() for x in data]
        except Exception:
            pass
    return ["cat", "dog"]


def update_prediction(history, labels, state, label_idx, confidence):
    if len(history) < WINDOW_SIZE:
        return state, label_idx, confidence
    votes = Counter(idx for idx, _ in history)
    majority, count = votes.most_common(1)[0]
    if count < MIN_MAJORITY:
        return state, label_idx, confidence
    confs = [c for idx, c in history if idx == majority]
    avg = float(np.mean(confs))
    if avg < CONFIDENCE_THRESHOLD:
        return "not_sure", majority, avg
    return labels[majority], int(majority), avg


def render(frame, info, show_saved):
    h, w = frame.shape[:2]
    title_h = max(40, int(h * 0.09))
    control_h = max(30, int(h * 0.07))

    pw = int(w * 0.75)
    px1 = (w - pw) // 2
    px2 = px1 + pw
    py1 = int(h * 0.13)
    py2 = int(h * 0.58)

    cv2.rectangle(frame, (0, 0), (w, title_h), C_TITLE_BG, -1)
    cv2.rectangle(frame, (0, h - control_h), (w, h), C_CONTROL_BG, -1)

    overlay = frame.copy()
    _rounded_filled(overlay, px1, py1, px2, py2, 18, C_PANEL_BG)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    _rounded_outline(frame, px1, py1, px2, py2, 18, C_BORDER, 2)

    ov = Overlay(frame)
    mid_x = w // 2

    ov.text(TITLE, mid_x, (title_h - 26) // 2, 26, C_TEXT, "center")
    ov.text("🤖 AI PREDICTION", mid_x, py1 + 8, 24, C_ACCENT, "center")

    state = info["state"]
    label = info["label"]
    conf = info["conf"]

    if state == "waiting":
        ov.text("🤔 WAITING...", mid_x, py1 + 38, 40, C_WARN, "center")
        ov.text("Show a cat or dog to the camera", mid_x, py1 + 108, 22, C_TEXT, "center")
        msg_lines = ["The AI will analyze the video in real time."]
    elif state == "not_sure":
        ov.text("🤔 NOT SURE", mid_x, py1 + 36, 42, C_WARN, "center")
        ov.text(f"Confidence: {int(conf * 100)}%", mid_x, py1 + 108, 26, C_TEXT, "center")
        msg_lines = ["Please show a clear cat or dog."]
    else:
        emoji = {"cat": "🐱", "dog": "🐶", "human": "🧑"}.get(state, "🤖")
        accent = {"cat": C_CAT, "dog": C_DOG, "human": C_HUMAN}.get(state, C_BORDER)
        ov.text(f"{emoji} {state.upper()} DETECTED!", mid_x, py1 + 36, 42, accent, "center")
        ov.text(f"Confidence: {int(conf * 100)}%", mid_x, py1 + 108, 26, C_TEXT, "center")
        cv2.line(frame, (px1 + 22, py1 + 140), (px2 - 22, py1 + 140), C_DIVIDER, 1)
        ov.text("🐾 FUN FACT", px1 + 26, py1 + 148, 20, C_ACCENT, "left")
        msg_lines = ov.wrap(FUN_FACTS.get(state, "The AI is watching."),
                            px2 - px1 - 52, 20)

    y_msg = py1 + 172
    for line in msg_lines[:2]:
        ov.text(line, px1 + 26, y_msg, 20, C_TEXT, "left")
        y_msg += 25

    if show_saved:
        ov.text("📸 Screenshot saved!", w - 18, title_h + 12, 24, C_SAVED, "right")

    ov.text("Q : Quit     R : Reset     S : Save Screenshot",
            mid_x, h - control_h + (control_h - 20) // 2, 20, C_TEXT, "center")

    return ov.finish()


def main():
    print("================================================")
    print(TITLE)
    print("================================================")
    print()
    print("Show your cat or dog to the camera!")
    print("The AI will try to identify it.")
    print()
    print("Press Q to quit.")
    print()

    model = load_ai_model()
    labels = load_class_labels()

    print("AI model loaded. Opening webcam...")
    print("Controls:  Q = Quit   |   R = Reset   |   S = Save Screenshot")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print()
        print("⚠️  Could not open webcam.")
        print()
        print("Check:")
        print("  - Camera connection")
        print("  - Camera permission")
        print("  - Another application using the camera")
        print()
        cap.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = "waiting"
    label_idx = None
    confidence = 0.0
    history = deque(maxlen=WINDOW_SIZE)
    frame_no = 0
    read_fails = 0
    saved_until = 0.0

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 800, 600)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_fails += 1
                if read_fails > 150:
                    print()
                    print("⚠️  Lost connection to the webcam.")
                    print("    Please reconnect the camera and run the program again.")
                    break
                time.sleep(0.03)
                continue
            read_fails = 0

            frame = cv2.flip(frame, 1)
            frame_no += 1

            if frame_no % PREDICT_EVERY == 0:
                try:
                    idx, conf = predict(model, frame)
                except Exception:
                    continue
                history.append((idx, conf))
                state, label_idx, confidence = update_prediction(
                    history, labels, state, label_idx, confidence
                )

            show_saved = time.time() < saved_until
            display = render(
                frame,
                {"state": state, "label": label_idx, "conf": confidence},
                show_saved,
            )

            cv2.imshow(TITLE, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                history.clear()
                state = "waiting"
                label_idx = None
                confidence = 0.0
                print()
                print("🤔 Reset. Waiting for an animal...")
            elif key in (ord("s"), ord("S")):
                shot = os.path.join(HERE, "captured_result.jpg")
                try:
                    cv2.imwrite(shot, display)
                    saved_until = time.time() + 2.0
                    print()
                    print("📸 Screenshot saved to:", shot)
                except Exception:
                    print()
                    print("⚠️  Could not save the screenshot.")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("Goodbye! Thanks for using the AI Cat vs Dog Detector.")


if __name__ == "__main__":
    main()