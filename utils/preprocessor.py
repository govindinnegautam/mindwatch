"""
STEP 3: Text Preprocessor
===========================
Cleans raw social media / user-generated text before feeding into BERT.

Run standalone test:
    python utils/preprocessor.py
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean raw user text for BERT input.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove special characters / HTML tags
        4. Collapse extra whitespace

    Args:
        text: raw string from user or social media

    Returns:
        cleaned string
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions and hashtags (social media)
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation except apostrophes (keeps "don't", "I'm" etc.)
    text = re.sub(r"[^\w\s']", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_keystroke_features(df, scaler=None, fit=True):
    """
    Normalize keystroke features using StandardScaler.
    Important: BERT embeddings don't need normalization,
    but keystroke values (speed=75, latency=0.09) are on very
    different scales — normalization prevents one feature dominating.

    Args:
        df    : DataFrame with typing_speed, keystroke_latency, error_rate
        scaler: existing scaler (pass during inference to use training stats)
        fit   : if True, fit a new scaler (use during training only)

    Returns:
        df with normalized features, scaler object
    """
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os

    feature_cols = ["typing_speed", "keystroke_latency", "error_rate"]

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/keystroke_scaler.pkl")
        print("✅ Scaler fitted and saved to models/keystroke_scaler.pkl")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I feel so TIRED and hopeless... http://t.co/abc #sad @user123",
        "Had a great day!! 😊 Check out www.example.com",
        "<b>I'm really struggling</b> with everything right now 123",
    ]
    print("=== Text Cleaning Test ===")
    for s in samples:
        print(f"  RAW  : {s}")
        print(f"  CLEAN: {clean_text(s)}")
        print()
