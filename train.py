"""
STEP 7: Training Script
=========================
Trains the full MindWatch multimodal pipeline.

What this script does:
    1. Loads and preprocesses the dataset
    2. Normalizes keystroke features
    3. Extracts BERT embeddings for all text samples
    4. Trains the fusion classifier
    5. Evaluates on test set and prints metrics matching Table 3 of paper
    6. Saves trained model weights

Run:
    python train.py

Expected output (approximate, matching paper Table 3):
    Accuracy : ~89%
    Precision: ~0.88
    Recall   : ~0.87
    F1-Score : ~0.87

Note for Colab users:
    Upload your project folder and run:
        !python train.py
    GPU will make BERT embedding ~10x faster.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.bert_model     import BERTFeatureExtractor
from models.fusion_model   import MindWatchClassifier
from utils.preprocessor    import clean_text

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_path":       "data/processed/dataset.csv",
    "model_save_path": "models/mindwatch_model.pth",
    "scaler_path":     "models/keystroke_scaler.pkl",
    "embeddings_path": "data/processed/bert_embeddings.pt",  # cache BERT output
    "test_size":       0.20,    # 80/20 split (matches paper Section 4.1)
    "batch_size":      16,
    "epochs":          10,
    "learning_rate":   2e-4,
    "dropout":         0.3,
    "seed":            42,
    "max_text_length": 128,
    "freeze_bert":     True,    # Set False to fine-tune BERT (needs GPU)
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# Use GPU if available (for Colab)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────

class MindWatchDataset(Dataset):
    """
    PyTorch Dataset that returns (text_embedding, behavior_features, label)
    for each sample.
    """

    def __init__(self, embeddings: torch.Tensor, behaviors: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.behaviors  = behaviors
        self.labels     = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],   # [768]
            self.behaviors[idx],    # [3]
            self.labels[idx],       # scalar
        )


# ─────────────────────────────────────────────────────────────
# Step 1: Load and Preprocess Data
# ─────────────────────────────────────────────────────────────

def load_data():
    print("\n📂 Step 1: Loading dataset...")

    if not os.path.exists(CONFIG["data_path"]):
        print("   Dataset not found. Generating it now...")
        os.system("python data/generate_dataset.py")

    df = pd.read_csv(CONFIG["data_path"])
    print(f"   Loaded {len(df)} samples | Labels: {df['label'].value_counts().to_dict()}")

    # Clean text
    print("   Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    return df


# ─────────────────────────────────────────────────────────────
# Step 2: Normalize Keystroke Features
# ─────────────────────────────────────────────────────────────

def normalize_features(df):
    print("\n📐 Step 2: Normalizing keystroke features...")

    feature_cols = ["typing_speed", "keystroke_latency", "error_rate"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"   ✅ Scaler saved to {CONFIG['scaler_path']}")

    return df, scaler


# ─────────────────────────────────────────────────────────────
# Step 3: Extract BERT Embeddings (with caching)
# ─────────────────────────────────────────────────────────────

def extract_bert_embeddings(df, bert_extractor):
    """
    Extracts [CLS] embeddings for all text samples.
    Caches to disk so you don't re-run BERT every training run.
    """
    cache_path = CONFIG["embeddings_path"]

    if os.path.exists(cache_path):
        print(f"\n⚡ Step 3: Loading cached BERT embeddings from {cache_path}")
        embeddings = torch.load(cache_path)
        print(f"   Loaded embeddings shape: {embeddings.shape}")
        return embeddings

    print(f"\n🤖 Step 3: Extracting BERT embeddings for {len(df)} samples...")
    print("   (This may take a few minutes — will be cached after first run)")

    embeddings = []
    bert_extractor.eval()
    bert_extractor.to(DEVICE)

    for i, text in enumerate(tqdm(df["text"].tolist(), desc="   BERT")):
        emb = bert_extractor.get_embedding(text, max_length=CONFIG["max_text_length"])
        embeddings.append(emb.cpu())

    embeddings = torch.stack(embeddings)  # [N, 768]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"   ✅ Embeddings cached to {cache_path}")
    print(f"   Embeddings shape: {embeddings.shape}")

    return embeddings


# ─────────────────────────────────────────────────────────────
# Step 4: Train the Model
# ─────────────────────────────────────────────────────────────

def train_model(train_loader, model):
    print(f"\n🏋️  Step 4: Training fusion classifier for {CONFIG['epochs']} epochs...")

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCELoss()

    # Learning rate scheduler — reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    history = {"loss": [], "accuracy": []}

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss   = 0.0
        correct      = 0
        total        = 0

        for text_emb, behavior, labels in train_loader:
            text_emb = text_emb.to(DEVICE)
            behavior = behavior.to(DEVICE)
            labels   = labels.to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(text_emb, behavior).squeeze()
            loss  = criterion(preds, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            predicted   = (preds >= 0.5).float()
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        scheduler.step(avg_loss)

        print(f"   Epoch [{epoch+1:2d}/{CONFIG['epochs']}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Accuracy: {accuracy:.1f}%")

    return model, history


# ─────────────────────────────────────────────────────────────
# Step 5: Evaluate and Print Metrics (Paper Table 3)
# ─────────────────────────────────────────────────────────────

def evaluate_model(test_loader, model):
    print("\n📊 Step 5: Evaluating on test set...")

    model.eval()
    all_preds  = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for text_emb, behavior, labels in test_loader:
            text_emb = text_emb.to(DEVICE)
            behavior = behavior.to(DEVICE)

            scores = model(text_emb, behavior).squeeze().cpu()
            preds  = (scores >= 0.5).float()

            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # ── Metrics (matching Table 3 of paper) ──────────────────────────────
    accuracy  = accuracy_score(all_labels, all_preds)  * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds,    zero_division=0)
    f1        = f1_score(all_labels, all_preds,        zero_division=0)
    auc       = roc_auc_score(all_labels, all_scores)

    print("\n" + "="*55)
    print("  📋 MindWatch — Results (Proposed Multimodal Framework)")
    print("="*55)
    print(f"  Accuracy  : {accuracy:.1f}%")
    print(f"  Precision : {precision:.2f}")
    print(f"  Recall    : {recall:.2f}")
    print(f"  F1-Score  : {f1:.2f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print("="*55)
    print("  📄 Paper reports: Accuracy=89.2%, Prec=0.88, Rec=0.87, F1=0.87")
    print("="*55)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "predictions": all_preds,
        "labels": all_labels,
        "scores": all_scores,
    }


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():
    print("🧠 MindWatch Training Pipeline")
    print("=" * 55)

    # 1. Load data
    df = load_data()

    # 2. Normalize keystroke features
    df, scaler = normalize_features(df)

    # 3. Extract BERT embeddings
    bert = BERTFeatureExtractor(freeze_bert=CONFIG["freeze_bert"])
    embeddings = extract_bert_embeddings(df, bert)

    # 4. Prepare tensors
    behaviors = torch.tensor(
        df[["typing_speed", "keystroke_latency", "error_rate"]].values,
        dtype=torch.float32
    )
    labels = torch.tensor(df["label"].values, dtype=torch.float32)

    # 5. Train/test split (80/20 — matches paper)
    indices     = list(range(len(df)))
    train_idx, test_idx = train_test_split(
        indices, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=labels
    )

    train_dataset = MindWatchDataset(
        embeddings[train_idx], behaviors[train_idx], labels[train_idx]
    )
    test_dataset  = MindWatchDataset(
        embeddings[test_idx],  behaviors[test_idx],  labels[test_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"\n   Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

    # 6. Train fusion model
    fusion_model = MindWatchClassifier(dropout_rate=CONFIG["dropout"])
    print(f"   Fusion model parameters: {fusion_model.count_parameters():,}")

    trained_model, history = train_model(train_loader, fusion_model)

    # 7. Evaluate
    results = evaluate_model(test_loader, trained_model)

    # 8. Save model
    torch.save(trained_model.state_dict(), CONFIG["model_save_path"])
    print(f"\n✅ Model saved to {CONFIG['model_save_path']}")
    print("\n🎉 Training complete! Run 'python evaluate.py' to generate plots.")


if __name__ == "__main__":
    main()
