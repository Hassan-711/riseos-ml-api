# RiseOS ML Microservice

## Setup

```bash
pip install -r requirements.txt
```

## Train Model

1. Download dataset:
   - https://archive.ics.uci.edu/ml/datasets/student+performance
   - Place `student-mat.csv` in this folder

2. Run training:
```bash
python train_model.py
```

This generates:
- `model.pkl` — trained GradientBoosting model
- `explainer.pkl` — SHAP TreeExplainer
- `feature_importance.json` — global feature rankings
- `accuracy.json` — model metrics

## Run API locally

```bash
uvicorn api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Deploy to Render (free)

1. Push this `/ml` folder to a GitHub repo
2. Go to render.com → New Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add env var: `PORT=10000`

Your API URL will be: `https://riseos-ml.onrender.com`

## Update RiseOS frontend

In `.env.local`:
```
NEXT_PUBLIC_ML_API_URL=https://riseos-ml.onrender.com
```
