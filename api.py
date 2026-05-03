"""
RiseOS ML Microservice — FastAPI
Serves predictions + SHAP explanations
Deploy on: Render / Railway (free tier)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
from typing import Optional

app = FastAPI(
    title="RiseOS AI Engine",
    description="Student Performance Prediction with Explainable AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts on startup
model    = joblib.load('model.pkl')
explainer = joblib.load('explainer.pkl')
with open('features.json') as f:
    FEATURES = json.load(f)
with open('feature_importance.json') as f:
    GLOBAL_IMPORTANCE = json.load(f)
with open('accuracy.json') as f:
    METRICS = json.load(f)

class StudentData(BaseModel):
    studytime: int   = Field(..., ge=1, le=4,  description="Weekly study time (1-4)")
    failures:  int   = Field(..., ge=0, le=4,  description="Past class failures")
    absences:  int   = Field(..., ge=0, le=93, description="Number of school absences")
    G1:        float = Field(..., ge=0, le=20, description="First period grade (0-20)")
    G2:        float = Field(..., ge=0, le=20, description="Second period grade (0-20)")
    health:    int   = Field(3,  ge=1, le=5,   description="Health status (1=bad, 5=excellent)")
    freetime:  int   = Field(3,  ge=1, le=5,   description="Free time after school")
    goout:     int   = Field(3,  ge=1, le=5,   description="Going out with friends")
    Dalc:      int   = Field(1,  ge=1, le=5,   description="Workday alcohol (1=low)")
    Walc:      int   = Field(1,  ge=1, le=5,   description="Weekend alcohol (1=low)")
    higher:    int   = Field(1,  ge=0, le=1,   description="Wants higher education")
    internet:  int   = Field(1,  ge=0, le=1,   description="Internet access at home")
    Medu:      int   = Field(2,  ge=0, le=4,   description="Mother's education level")
    Fedu:      int   = Field(2,  ge=0, le=4,   description="Father's education level")

class PredictionResponse(BaseModel):
    prediction:   str
    confidence:   float
    probabilities: dict
    shap_explanation: list
    risk_level:   str
    insights:     list
    model_accuracy: float

@app.get("/health")
def health():
    return {"status": "ok", "model_accuracy": METRICS['accuracy']}

@app.get("/global-importance")
def global_importance():
    return {"features": GLOBAL_IMPORTANCE, "model_accuracy": METRICS['accuracy']}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    try:
        # Build input array
        input_data = np.array([[
            data.studytime, data.failures, data.absences,
            data.G1, data.G2, data.health, data.freetime,
            data.goout, data.Dalc, data.Walc,
            data.higher, data.internet, data.Medu, data.Fedu
        ]])

        # Predict
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        classes    = model.classes_.tolist()
        confidence = float(max(pred_proba))
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, pred_proba)}

        # SHAP local explanation
        shap_vals = explainer.shap_values(input_data)
        # For multi-class, take the predicted class index
        pred_idx = classes.index(pred_class)
        if isinstance(shap_vals, list):
            local_shap = shap_vals[pred_idx][0]
        elif shap_vals.ndim == 3:
            # Fix for SHAP 0.50.0+ (samples, features, classes)
            local_shap = shap_vals[0, :, pred_idx]
        else:
            local_shap = shap_vals[0]

        # Format SHAP as ranked list
        shap_items = sorted(
            [{"feature": f, "value": float(v), "impact": "positive" if v > 0 else "negative"}
             for f, v in zip(FEATURES, local_shap)],
            key=lambda x: abs(x["value"]), reverse=True
        )

        # Risk level
        risk_map = {"Fail": "high", "Pass": "medium", "Good": "low", "Excellent": "low"}
        risk_level = risk_map.get(pred_class, "medium")

        # Human-readable insights
        insights = generate_insights(data, pred_class, local_shap)

        return PredictionResponse(
            prediction=pred_class,
            confidence=round(confidence, 4),
            probabilities=proba_dict,
            shap_explanation=shap_items[:6],  # top 6 factors
            risk_level=risk_level,
            insights=insights,
            model_accuracy=METRICS['accuracy']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_insights(data: StudentData, prediction: str, shap_vals: np.ndarray) -> list:
    """Generate human-readable explanations"""
    insights = []

    if data.G1 > 0 and data.G2 > 0:
        trend = data.G2 - data.G1
        if trend > 2:
            insights.append("📈 Your grades are improving across periods — great momentum!")
        elif trend < -2:
            insights.append("📉 Grades dropped between periods — focus needed.")

    if data.studytime <= 1:
        insights.append("⚠️ Study time is very low (< 2 hrs/week). Increasing it significantly improves outcomes.")
    elif data.studytime >= 3:
        insights.append("✅ Good study habits — consistent study time is your biggest asset.")

    if data.failures > 0:
        insights.append(f"⚠️ {data.failures} past failure(s) detected — extra attention needed in weak subjects.")

    if data.absences > 15:
        insights.append(f"🔴 High absences ({data.absences}) — attendance directly impacts final results.")
    elif data.absences <= 3:
        insights.append("✅ Excellent attendance — consistent presence is a key success factor.")

    if prediction == "Fail":
        insights.append("🎯 Focus: Prioritize catching up on syllabus. Consider reducing goout frequency.")
    elif prediction == "Excellent":
        insights.append("🏆 On track for excellent results. Maintain your current habits.")

    return insights[:4]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
