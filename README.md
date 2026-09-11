# 🐱🐶🧑 Cat / Dog / Human Detector (Stem Detection)

Real-time cats vs dogs vs humans classifier deployed as a web app.

## Architecture

- `backend/` — FastAPI + TFLite (MobileNetV2) REST API — **deployed on Render (free)**
- `frontend/` — static webcam web app (HTML/CSS/JS) — **deployed on Vercel (free)**
- `cat-dog-camera/` — original local project: training, dataset (cat/dog/human, 400 each), webcam GUI

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`          | API info + class names |
| GET    | `/health`    | Server + model status |
| POST   | `/predict`   | `{ "image": "<base64 jpeg>" }` → prediction, confidence, fun fact |

Example:

```bash
curl -X POST https://<backend>/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64 data>"}'
```

## Live URLs

- **Frontend:** deployed to Vercel (see `frontend/vercel.json` rewrites to backend).
- **Backend:** deployed to Render via `render.yaml` blueprint (free plan).

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