"""
STEP 10: Single Prediction Script
===================================
Run a prediction directly from command line — no API needed.

Usage:
    python predict.py

Or with custom input:
    python predict.py --text "I feel exhausted" --speed 42 --latency 0.22 --errors 0.10
"""

import os
import sys
import torch
import numpy as np
import joblib
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.bert_model   import BERTFeatureExtractor
from models.fusion_model import MindWatchClassifier
from utils.preprocessor  import clean_text


def load_model():
    """Load trained model and scaler."""
    if not os.path.exists("models/mindwatch_model.pth"):
        print("❌ Model not found. Please run 'python train.py' first.")
        sys.exit(1)

    bert   = BERTFeatureExtractor(freeze_bert=True)
    bert.eval()

    model  = MindWatchClassifier()
    model.load_state_dict(torch.load("models/mindwatch_model.pth", map_location="cpu"))
    model.eval()

    scaler = joblib.load("models/keystroke_scaler.pkl")

    return bert, model, scaler


def predict(text: str, typing_speed: float, keystroke_latency: float, error_rate: float):
    """Full prediction pipeline for a single input."""

    bert, model, scaler = load_model()

    # 1. Clean and embed text
    cleaned        = clean_text(text)
    text_embedding = bert.get_embedding(cleaned)

    # 2. Normalize keystroke features
    raw      = np.array([[typing_speed, keystroke_latency, error_rate]])
    norm     = scaler.transform(raw)
    behavior = torch.tensor(norm[0], dtype=torch.float32)

    # 3. Predict
    result = model.predict_risk(text_embedding, behavior)

    # 4. Display
    print("\n" + "="*50)
    print("  🧠 MindWatch Prediction Result")
    print("="*50)
    print(f"  Input text   : {text[:60]}...")
    print(f"  Typing speed : {typing_speed} keys/sec")
    print(f"  Latency      : {keystroke_latency} sec")
    print(f"  Error rate   : {error_rate}")
    print("-"*50)
    print(f"  Risk Score   : {result['risk_score']:.4f}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Confidence   : {result['confidence']:.4f}")
    print("="*50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindWatch single prediction")
    parser.add_argument("--text",    type=str,   default="I feel so exhausted and everything seems hopeless lately")
    parser.add_argument("--speed",   type=float, default=42.0)
    parser.add_argument("--latency", type=float, default=0.22)
    parser.add_argument("--errors",  type=float, default=0.10)
    args = parser.parse_args()

    predict(
        text              = args.text,
        typing_speed      = args.speed,
        keystroke_latency = args.latency,
        error_rate        = args.errors,
    )
