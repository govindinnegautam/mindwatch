"""
STEP 4: BERT Text Feature Extractor
=====================================
Implements Section 3.2 of the MindWatch paper.

Key equations from paper:
    X = {x1, x2, ..., xn}        (Equation 1 - input text sequence)
    T = BERT(X)                   (Equation 2 - contextual embedding)
    T ∈ R^d  where d = 768        ([CLS] token hidden state)

The [CLS] (classification) token captures the full sentence meaning.
We fine-tune BERT specifically for mental health text classification.

Usage:
    from models.bert_model import BERTFeatureExtractor
    extractor = BERTFeatureExtractor()
    embedding = extractor.get_embedding("I feel really anxious today")
    # embedding.shape → torch.Size([768])
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os


class BERTFeatureExtractor(nn.Module):
    """
    Wraps BERT-base-uncased to extract [CLS] token embeddings.
    Includes a fine-tuning head that will be trained on our dataset.
    """

    def __init__(self, model_name: str = "bert-base-uncased", freeze_bert: bool = False):
        """
        Args:
            model_name  : HuggingFace model identifier
            freeze_bert : If True, only train the classification head
                          (faster, use when dataset is small)
        """
        super().__init__()

        print(f"⏳ Loading {model_name} tokenizer and model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert      = BertModel.from_pretrained(model_name)

        if freeze_bert:
            # Freeze all BERT weights — only train the head
            for param in self.bert.parameters():
                param.requires_grad = False
            print("   ℹ️  BERT weights frozen (fine-tuning head only)")

        # Hidden dim from BERT-base = 768
        self.hidden_dim = self.bert.config.hidden_size  # 768
        print(f"✅ BERT loaded | hidden_dim = {self.hidden_dim}")

    def forward(self, input_ids, attention_mask):
        """
        Full forward pass — used during training.

        Args:
            input_ids      : tokenized input tensor  [batch, seq_len]
            attention_mask : mask tensor              [batch, seq_len]

        Returns:
            cls_embedding: [batch, 768]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # last_hidden_state[:, 0, :] = [CLS] token = sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        return cls_embedding

    def get_embedding(self, text: str, max_length: int = 128) -> torch.Tensor:
        """
        Convenience method for inference on a single string.

        Args:
            text      : raw input string
            max_length: max tokens (paper uses short social media text)

        Returns:
            embedding tensor of shape [768]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            embedding = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        return embedding.squeeze(0)  # remove batch dim → [768]

    def tokenize_batch(self, texts: list, max_length: int = 128) -> dict:
        """
        Tokenize a list of texts for batch processing during training.

        Args:
            texts     : list of raw strings
            max_length: max token length

        Returns:
            dict with input_ids and attention_mask tensors
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = BERTFeatureExtractor(freeze_bert=True)

    test_texts = [
        "I feel completely hopeless and exhausted all the time",
        "Had an amazing day, feeling really positive and grateful",
    ]

    print("\n=== Embedding Test ===")
    for text in test_texts:
        emb = extractor.get_embedding(text)
        print(f"  Text : {text[:50]}...")
        print(f"  Shape: {emb.shape}  | First 5 values: {emb[:5].tolist()}\n")
