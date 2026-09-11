# 🐱🐶🧑 Cat / Dog / Human Detector (Stem Detection)

Real-time cats vs dogs vs humans classifier deployed as a web app.

## Architecture

- `backend/` — FastAPI + LiteRT (MobileNetV2 TFLite) REST API — **deployed on Vercel (free)**
- `frontend/` — static webcam web app (HTML/CSS/JS) — **deployed on Vercel (free)**
- `cat-dog-camera/` — original local project: training, dataset (cat/dog/human, 400 each), webcam GUI

## Live URLs

- **App (site + API proxy):** https://stem-detection.vercel.app
- **API:** https://stem-detection-api.vercel.app

The frontend proxies `/api/*` to the backend project via `frontend/vercel.json` rewrites, so the app uses a single same-origin URL. No card / paid plan required.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api`          | API info + class names |
| GET    | `/api/health`   | Server + model status |
| POST   | `/api/predict`  | `{ "image": "<base64 jpeg>" }` → prediction, confidence, fun fact |

Example:

```bash
curl -X POST https://stem-detection-api.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64 data>"}'
```

## Deploying

Backend (fresh Vercel project — framework auto-detects FastAPI from `app.py`):

```bash
cd backend
npm i -g vercel
vercel deploy --prod --yes
```

Frontend (static site; rewrites `/api/*` to the backend URL):

```bash
cd frontend
vercel deploy --prod --yes
```

Notes:
- PyPI `tflite-runtime` only ships wheels up to cp311, but Vercel runs Python 3.12 — use `ai-edge-litert` (the official LiteRT successor) instead, which has cp312 wheels.
- Render was not used: its API requires payment info even on the free plan.

## Local development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

Frontend dev: serve `frontend/` with any static server; `API_BASE` in `frontend/config.js` defaults to the Vercel `/api` proxy.

## Training

See `cat-dog-camera/` → `train_model.py` (MobileNetV2 fine-tune on ~1200 images).