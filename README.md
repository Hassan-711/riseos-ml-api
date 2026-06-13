# 🧠 RiseOS ML Microservice

This repository contains the intelligent backend engine powering **RiseOS**. Built on FastAPI, this microservice handles Machine Learning predictions, Explainable AI (XAI) processing, Generative AI prompt engineering, and Markdown report generation.

*Developed as part of the Non-Teaching Credit Course (NTCC) academic project.*

## 🛠 Tech Stack
* **Framework:** FastAPI (Python), Uvicorn
* **Generative AI:** Google Gemini API (`gemini-2.5-flash`)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), NumPy, Pandas
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Deployment:** Render (Web Service)

## ✨ Core API Modules

### 1. Generative AI Engine (`/generate-roadmap`)
* Receives a user goal and timeframe.
* Uses precise prompt engineering with Google Gemini to generate granular, day-by-day learning milestones in strict, parseable JSON format.

### 2. Explainable AI Academic Predictor (`/predict`)
* Uses a pre-trained **Random Forest Classifier** (`model.pkl`) to predict student success probabilities based on 14 real-world parameters (study time, attendance, past failures, etc.).
* Utilizes a SHAP TreeExplainer (`explainer.pkl`) to decode the model's logic, returning exact feature impacts (e.g., "Why did the model predict a fail?").
* Generates human-readable, actionable insights based on local SHAP values to guide student improvement.

### 3. Utility Services (`/export-roadmap` & `/health`)
* Processes complex, nested JSON data from the frontend to dynamically format and return downloadable Markdown (`.md`) files for easy roadmap sharing.
* Provides health checks and serves global feature importance metrics to the frontend dashboard.

## ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ml

2. **Install dependencies:**
   pip install -r requirements.txt

3. **Set Environment Variables:**
   export GEMINI_API_KEY="your_google_gemini_api_key"

4. **Start the Uvicorn Server:**
   uvicorn api:app --reload --port 8000
Access the interactive Swagger API documentation at http://localhost:8000/docs.

## 📊 Model Training (Optional)
If you wish to retrain the underlying Random Forest model and regenerate the SHAP explainers:

Ensure the UCI dataset student-mat.csv is in the root directory.

Run the training script:
   python train_model.py

This will automatically generate and save fresh model.pkl, explainer.pkl, feature_importance.json, and accuracy.json artifacts.


## 📂 Project Structure

```text
riseos-ml-api/
├── api.py                   # Main FastAPI application & API routes
├── train_model.py           # Script to train Random Forest & generate SHAP explainer
├── student-mat.csv          # UCI Student Performance Dataset
├── requirements.txt         # Python dependencies
├── model.pkl                # Pre-trained Random Forest model
├── explainer.pkl            # Pre-trained SHAP explainer artifact
├── features.json            # Cached feature names
├── feature_importance.json  # Global SHAP feature importance metrics
├── accuracy.json            # Model evaluation metrics
└── README.md                # Project documentation

## ☁️ Deployment to Render (Free Tier)
Push this /ml folder to a GitHub repository.

Go to your Render dashboard and create a New Web Service.

Build Command: pip install -r requirements.txt

Start Command: uvicorn api:app --host 0.0.0.0 --port $PORT

Add your GEMINI_API_KEY to the Environment Variables section in Render.

Copy your live Render URL and add it to your frontend's .env.local as NEXT_PUBLIC_ML_API_URL.