"""
STEP 6: Multimodal Fusion Classifier
======================================
Implements Sections 3.4 and 3.6 of the MindWatch paper.

Paper equations implemented:
    F = [T || B]                     (Equation 4 - feature concatenation)
    P(y=1|F) = σ(WF + b)            (Equation 5 - risk classification)

Architecture:
    Text embedding T     : [batch, 768]   ← from BERT
    Behavioral vector B  : [batch,   3]   ← from keystroke extractor
    Fused vector F       : [batch, 771]   ← concatenation [T || B]
    Dense layer          : [batch, 256]   ← feature interaction learning
    Output               : [batch,   1]   ← risk probability [0, 1]

Usage:
    from models.fusion_model import MindWatchClassifier
    model = MindWatchClassifier()
    risk = model(text_embedding, behavior_features)  # tensor([0.82])
"""

import torch
import torch.nn as nn
import numpy as np


class MindWatchClassifier(nn.Module):
    """
    Multimodal mental health risk classifier.

    Fuses BERT text embeddings with keystroke behavioral features
    and outputs a risk probability score between 0 and 1.

    Score interpretation:
        0.0 – 0.4  → LOW risk
        0.4 – 0.6  → MODERATE risk (borderline)
        0.6 – 1.0  → HIGH risk
    """

    def __init__(
        self,
        text_dim:     int   = 768,   # BERT-base hidden size
        behavior_dim: int   = 3,     # [speed, latency, error_rate]
        hidden_dim:   int   = 256,   # dense layer width
        dropout_rate: float = 0.3,   # regularization
    ):
        super().__init__()

        self.text_dim     = text_dim
        self.behavior_dim = behavior_dim
        fused_dim         = text_dim + behavior_dim  # 771

        # ── Classification head ─────────────────────────────────────────────
        # Implements the Dense Layer + Sigmoid from Figure 3 of the paper
        self.classifier = nn.Sequential(
            # Layer 1: Fusion projection
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: Risk estimation
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output: binary risk probability
            nn.Linear(64, 1),
            nn.Sigmoid(),  # σ(WF + b) → output ∈ [0, 1]
        )

        # Initialize weights using Xavier uniform for stable training
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for faster convergence."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        text_embedding:     torch.Tensor,  # [batch, 768]
        behavior_features:  torch.Tensor,  # [batch, 3]
    ) -> torch.Tensor:
        """
        Forward pass implementing F = [T || B] then P = σ(WF + b).

        Args:
            text_embedding    : BERT [CLS] token, shape [batch, 768]
            behavior_features : keystroke vector, shape [batch, 3]

        Returns:
            risk_probability: shape [batch, 1], values in [0, 1]
        """
        # Ensure both tensors are float32
        text_embedding    = text_embedding.float()
        behavior_features = behavior_features.float()

        # Handle 1D tensors (single sample inference)
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)      # [1, 768]
        if behavior_features.dim() == 1:
            behavior_features = behavior_features.unsqueeze(0)  # [1, 3]

        # Equation 4: F = [T || B]  (concatenate along feature dimension)
        fused = torch.cat([text_embedding, behavior_features], dim=-1)  # [batch, 771]

        # Equation 5: P(y=1|F) = σ(WF + b)
        risk_probability = self.classifier(fused)  # [batch, 1]

        return risk_probability

    def predict_risk(
        self,
        text_embedding:    torch.Tensor,
        behavior_features: torch.Tensor,
        threshold:         float = 0.5,
    ) -> dict:
        """
        Predict risk with human-readable output. Use during inference.

        Args:
            text_embedding    : BERT embedding [768]
            behavior_features : keystroke vector [3]
            threshold         : decision boundary (default 0.5)

        Returns:
            dict with score, label, and risk_level
        """
        self.eval()
        with torch.no_grad():
            prob  = self.forward(text_embedding, behavior_features)
            score = prob.item()

        label = 1 if score >= threshold else 0

        if score < 0.4:
            risk_level = "LOW"
        elif score < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        return {
            "risk_score":  round(score, 4),
            "risk_label":  label,
            "risk_level":  risk_level,
            "confidence":  round(abs(score - 0.5) * 2, 4),  # 0=uncertain, 1=confident
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Fusion Model Test ===\n")

    model = MindWatchClassifier()
    print(f"  Trainable parameters: {model.count_parameters():,}")
    print(f"  Input  : text_dim={model.text_dim} + behavior_dim={model.behavior_dim} = {model.text_dim + model.behavior_dim}")
    print(f"  Output : risk probability ∈ [0, 1]\n")

    # Simulate a batch of 4 samples
    batch_text     = torch.randn(4, 768)
    batch_behavior = torch.tensor([
        [42.0, 0.22, 0.10],   # at-risk-like
        [75.0, 0.09, 0.02],   # healthy-like
        [35.0, 0.28, 0.15],   # at-risk-like
        [80.0, 0.08, 0.01],   # healthy-like
    ])

    output = model(batch_text, batch_behavior)
    print(f"  Batch output shape: {output.shape}")
    print(f"  Risk scores: {output.squeeze().tolist()}\n")

    # Single sample inference
    single_text     = torch.randn(768)
    single_behavior = torch.tensor([40.0, 0.25, 0.12])
    result          = model.predict_risk(single_text, single_behavior)
    print(f"  Single inference result: {result}")
