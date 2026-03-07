"""
STEP 9: FastAPI Backend
========================
REST API that serves the trained MindWatch model.

Endpoints:
    GET  /              → health check
    POST /predict       → predict risk from text + keystroke data
    GET  /demo          → returns a sample prediction for testing

Run:
    uvicorn api.app:app --reload --port 8000

Test via browser:
    http://localhost:8000/docs   ← Swagger UI (auto-generated!)

Test via curl:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text":"I feel hopeless","typing_speed":42.0,"keystroke_latency":0.22,"error_rate":0.10}'
"""

import os
import sys
import torch
import numpy as np
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bert_model   import BERTFeatureExtractor
from models.fusion_model import MindWatchClassifier
from utils.preprocessor  import clean_text

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    
    title       = "MindWatch API",
    description = "Mental health risk prediction using multimodal AI (text + typing behavior)",
    version     = "1.0.0",
)

# Serve the web frontend
static_path = os.path.join(os.path.dirname(__file__), "../static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(static_path, "index.html"))

# Allow requests from any origin (needed for web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────
# Load model on startup
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These will be populated on startup
bert_extractor   = None
fusion_model     = None
keystroke_scaler = None


@app.on_event("startup")
async def load_models():
    """Load all model artifacts when the server starts."""
    global bert_extractor, fusion_model, keystroke_scaler

    print("⏳ Loading MindWatch models...")

    try:
        # BERT extractor
        bert_extractor = BERTFeatureExtractor(freeze_bert=True)
        bert_extractor.to(DEVICE)
        bert_extractor.eval()

        # Fusion model
        fusion_model = MindWatchClassifier()
        model_path   = os.path.join(os.path.dirname(__file__), "../models/mindwatch_model.pth")
        fusion_model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        fusion_model.to(DEVICE)
        fusion_model.eval()

        # Keystroke scaler
        scaler_path      = os.path.join(os.path.dirname(__file__), "../models/keystroke_scaler.pkl")
        keystroke_scaler = joblib.load(scaler_path)

        print("✅ All models loaded successfully!")

    except FileNotFoundError as e:
        print(f"⚠️  Model files not found: {e}")
        print("   Please run 'python train.py' first to train the model.")


# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    text:               str   = Field(...,  description="User's text input", example="I feel really exhausted and hopeless lately")
    typing_speed:       float = Field(...,  description="Keystrokes per second",          example=42.3,  ge=0)
    keystroke_latency:  float = Field(...,  description="Average inter-key delay (secs)", example=0.22,  ge=0)
    error_rate:         float = Field(...,  description="Ratio of correction keystrokes", example=0.09,  ge=0, le=1)


class PredictionResponse(BaseModel):
    risk_score:       float
    risk_level:       str    # LOW / MODERATE / HIGH
    risk_label:       int    # 0 or 1
    confidence:       float
    cleaned_text:     str
    interpretation:   str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {
        "status":  "running",
        "model":   "MindWatch v1.0",
        "device":  str(DEVICE),
        "ready":   fusion_model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict mental health risk")
def predict(req: PredictionRequest):
    """
    Predicts mental health risk from text + typing behavior.

    Returns a risk score between 0 (no risk) and 1 (high risk),
    along with a risk level (LOW / MODERATE / HIGH).
    """
    if fusion_model is None or bert_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train the model first using train.py"
        )

    # 1. Clean text
    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text input is empty after cleaning")

    # 2. Get BERT embedding
    text_embedding = bert_extractor.get_embedding(cleaned)  # [768]

    # 3. Normalize keystroke features using saved scaler
    raw_keystroke = np.array([[
        req.typing_speed,
        req.keystroke_latency,
        req.error_rate
    ]])
    normalized_keystroke = keystroke_scaler.transform(raw_keystroke)
    behavior_tensor      = torch.tensor(normalized_keystroke[0], dtype=torch.float32)

    # 4. Fuse and predict
    result = fusion_model.predict_risk(text_embedding, behavior_tensor)
    score  = result["risk_score"]

    # 5. Human-readable interpretation
    if result["risk_level"] == "HIGH":
        interpretation = (
            "⚠️  Elevated mental health risk indicators detected. "
            "Consider speaking with a mental health professional."
        )
    elif result["risk_level"] == "MODERATE":
        interpretation = (
            "🟡 Some stress indicators present. "
            "Practice self-care and monitor your wellbeing."
        )
    else:
        interpretation = (
            "✅ Low risk indicators. "
            "Keep maintaining healthy habits and routines."
        )

    return PredictionResponse(
        risk_score     = result["risk_score"],
        risk_level     = result["risk_level"],
        risk_label     = result["risk_label"],
        confidence     = result["confidence"],
        cleaned_text   = cleaned,
        interpretation = interpretation,
    )


@app.get("/demo", summary="Demo prediction with sample data")
def demo():
    """Returns two example predictions for testing the API without a frontend."""
    if fusion_model is None:
        return {"error": "Model not loaded. Run train.py first."}

    samples = [
        {
            "label"   : "At-risk sample",
            "request" : PredictionRequest(
                text              = "I feel so empty and hopeless, nothing matters anymore",
                typing_speed      = 38.0,
                keystroke_latency = 0.25,
                error_rate        = 0.12,
            )
        },
        {
            "label"   : "Healthy sample",
            "request" : PredictionRequest(
                text              = "Had a great productive day, feeling really positive",
                typing_speed      = 78.0,
                keystroke_latency = 0.09,
                error_rate        = 0.02,
            )
        },
    ]

    results = []
    for sample in samples:
        result = predict(sample["request"])
        results.append({
            "sample": sample["label"],
            "result": result
        })

    return {"demo_results": results}
