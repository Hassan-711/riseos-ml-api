"""
RiseOS ML Microservice — FastAPI
Serves predictions + SHAP explanations + Gemini AI Roadmap Generator
"""

import os
import json
import joblib
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="RiseOS AI Engine",
    description="Student Performance Prediction with Explainable AI & Task Generator",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. AI ROADMAP GENERATOR SETUP (GEMINI) ──────────────────────────────────
from google import genai # <--- NAYA IMPORT
from google.genai import types
import os
import json
from pydantic import BaseModel, Field
from fastapi import HTTPException

# Yahan par apni wahi copy ki hui key daal dena (quotes ke andar)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Naya client initialize karna
client = genai.Client(api_key=GEMINI_API_KEY)

class RoadmapRequest(BaseModel):
    prompt: str = Field(..., description="E.g., Master System Design in 30 days")
    days: int = Field(30, description="Number of days for the roadmap")

class RoadmapResponse(BaseModel):
    tasks: list

@app.post("/generate-roadmap", response_model=RoadmapResponse)
def generate_roadmap(data: RoadmapRequest):
    try:
        system_prompt = f"""
        You are an expert technical mentor and productivity coach.
        The user wants to achieve this goal: "{data.prompt}" in exactly {data.days} days.
        
        Generate a day-by-day actionable task list. Break down complex topics into bite-sized daily goals.
        
        IMPORTANT RULES:
        1. Respond ONLY with a valid JSON array.
        2. Do NOT wrap the JSON in markdown blocks (like ```json ... ```).
        3. Do NOT add any introductory or concluding text.
        4. Each object in the array must have exactly two keys: "title" and "priority".
        5. "title" must be a string starting with "Day X: [Task Name]". Keep it under 60 characters.
        6. "priority" must be exactly one of these lowercase strings: "high", "medium", "low", "urgent".

        Example output format:
        [
          {{"title": "Day 1: Understand Client-Server Architecture", "priority": "high"}},
          {{"title": "Day 2: Learn about Load Balancing", "priority": "medium"}}
        ]
        """
        
        # Naye package ka function use karna
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Hum latest aur fastest model use kar rahe hain
            contents=system_prompt,
        )
        
        raw_text = response.text.strip()
        # Clean up Markdown formatting just in case
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        raw_text = raw_text.strip()
        tasks_list = json.loads(raw_text)
        
        return {"tasks": tasks_list}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI failed to generate valid JSON format. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. ML MODEL SETUP (STUDENT PREDICTION) ──────────────────────────────────
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
        pred_idx = classes.index(pred_class)
        if isinstance(shap_vals, list):
            local_shap = shap_vals[pred_idx][0]
        elif shap_vals.ndim == 3:
            local_shap = shap_vals[0, :, pred_idx]
        else:
            local_shap = shap_vals[0]

        # Format SHAP as ranked list
        shap_items = sorted(
            [{"feature": f, "value": float(v), "impact": "positive" if v > 0 else "negative"}
             for f, v in zip(FEATURES, local_shap)],
            key=lambda x: abs(x["value"]), reverse=True
        )

        risk_map = {"Fail": "high", "Pass": "medium", "Good": "low", "Excellent": "low"}
        risk_level = risk_map.get(pred_class, "medium")

        insights = generate_insights(data, pred_class, local_shap)

        return PredictionResponse(
            prediction=pred_class,
            confidence=round(confidence, 4),
            probabilities=proba_dict,
            shap_explanation=shap_items[:6],
            risk_level=risk_level,
            insights=insights,
            model_accuracy=METRICS['accuracy']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_insights(data: StudentData, prediction: str, shap_vals: np.ndarray) -> list:
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