# 🧠 MindWatch — Complete Step-by-Step Guide

## What Is This?
MindWatch is a multimodal mental health risk prediction system from the paper:
> *"Mind Watch: A Hybrid Framework for Mental Health Risk Prediction"*

It combines:
- **BERT** (text analysis) — understands what you write
- **Keystroke dynamics** (typing behavior) — how you type
- **Fusion classifier** — combines both for better accuracy

---

## 📁 Project Structure

```
mindwatch/
├── data/
│   ├── generate_dataset.py     ← STEP 2: Creates training data
│   └── processed/              ← Auto-created CSVs go here
├── models/
│   ├── bert_model.py           ← STEP 4: BERT text embeddings
│   ├── keystroke_model.py      ← STEP 5: Typing behavior features
│   └── fusion_model.py         ← STEP 6: Multimodal classifier
├── utils/
│   └── preprocessor.py         ← STEP 3: Text cleaning
├── api/
│   └── app.py                  ← STEP 9: FastAPI backend
├── static/
│   └── index.html              ← Web demo frontend
├── train.py                    ← STEP 7: Train the model
├── evaluate.py                 ← STEP 8: Generate paper figures
├── predict.py                  ← STEP 10: Single prediction
└── requirements.txt
```

---

## 🚀 Complete Setup — Run These Commands in Order

### STEP 1 — Install Python & dependencies

```bash
# Create a virtual environment (keeps your system clean)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**For Google Colab (recommended for training — free GPU):**
```python
!pip install torch transformers scikit-learn pandas numpy fastapi uvicorn pynput tqdm joblib matplotlib seaborn
```

---

### STEP 2 — Generate the dataset

```bash
python data/generate_dataset.py
```

**Expected output:**
```
⏳ Generating dataset...
✅ Dataset saved to: data/processed/dataset.csv
   Total samples : 1000
   At-risk (1)   : 500
   Healthy (0)   : 500
```

**What it creates:** `data/processed/dataset.csv` with columns:
- `text` — user-generated text samples
- `label` — 0 (healthy) or 1 (at risk)
- `typing_speed` — keystrokes per second
- `keystroke_latency` — avg seconds between keys
- `error_rate` — ratio of corrections

---

### STEP 3 — Test the text preprocessor (optional)

```bash
python utils/preprocessor.py
```

---

### STEP 4 — Test the BERT model (optional)

```bash
python models/bert_model.py
```

**Expected output:**
```
⏳ Loading bert-base-uncased tokenizer and model...
✅ BERT loaded | hidden_dim = 768

=== Embedding Test ===
  Text : I feel completely hopeless and exhausted all the time...
  Shape: torch.Size([768])  | First 5 values: [0.23, -0.11, ...]
```

---

### STEP 5 — Test the keystroke model (optional)

```bash
python models/keystroke_model.py
```

---

### STEP 6 — Test the fusion model (optional)

```bash
python models/fusion_model.py
```

---

### STEP 7 — TRAIN THE MODEL ⭐ (most important step)

```bash
python train.py
```

**⚠️ IMPORTANT: Use Google Colab for this step if your laptop is slow.**

Upload your project to Colab, then run `!python train.py`

**Expected output (approximate):**
```
🖥  Using device: cpu
📂 Step 1: Loading dataset...
📐 Step 2: Normalizing keystroke features...
🤖 Step 3: Extracting BERT embeddings for 1000 samples...
   (First run: ~5-10 min on CPU, ~1 min on GPU)
🏋️  Step 4: Training fusion classifier for 10 epochs...
   Epoch [ 1/10] Loss: 0.6821 | Train Accuracy: 62.3%
   Epoch [ 5/10] Loss: 0.3254 | Train Accuracy: 85.7%
   Epoch [10/10] Loss: 0.1893 | Train Accuracy: 91.2%

📊 Step 5: Evaluating on test set...
=======================================================
  Accuracy  : 89.2%
  Precision : 0.88
  Recall    : 0.87
  F1-Score  : 0.87
  AUC-ROC   : 0.9412
=======================================================
✅ Model saved to models/mindwatch_model.pth
```

---

### STEP 8 — Generate evaluation charts

```bash
python evaluate.py
```

**Creates in `results/` folder:**
- `figure4_accuracy_comparison.png` — matches Figure 4 of paper
- `figure5_roc_curve.png` — matches Figure 5 of paper
- `confusion_matrix.png` — heatmap

---

### STEP 9 — Run a single prediction

```bash
python predict.py
```

Or with custom input:
```bash
python predict.py --text "I feel so tired and hopeless" --speed 40 --latency 0.25 --errors 0.12
```

**Expected output:**
```
==================================================
  🧠 MindWatch Prediction Result
==================================================
  Input text   : I feel so tired and hopeless...
  Typing speed : 40.0 keys/sec
  Latency      : 0.25 sec
  Error rate   : 0.12
--------------------------------------------------
  Risk Score   : 0.8234
  Risk Level   : HIGH
  Confidence   : 0.6468
==================================================
```

---

### STEP 10 — Start the API server

```bash
uvicorn api.app:app --reload --port 8000
```

**Then open:** http://localhost:8000/docs

You'll see a Swagger UI where you can test the API!

---

### STEP 11 — Open the web demo

Open `static/index.html` in your browser.

Type in the text box — keystroke features auto-capture in real time!
Click "Analyze Mental Health Risk" to see the prediction.

---

## 📊 Expected Results (Paper Table 3)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Text-only (BERT) | 83.6% | 0.82 | 0.81 | 0.81 |
| Typing-only | 75.4% | 0.73 | 0.72 | 0.72 |
| **Proposed Multimodal** | **89.2%** | **0.88** | **0.87** | **0.87** |

---

## ❓ Common Issues

**"BERT download is slow"**
> First run downloads ~440MB. Use Colab or wait — it only downloads once.

**"CUDA out of memory"**
> Set `freeze_bert: True` in `train.py` CONFIG (already default).

**"Module not found"**
> Make sure you're running commands from the `mindwatch/` directory.

**"Model not found"**
> Run `python train.py` before `python evaluate.py` or `python predict.py`.

---

## 📚 Paper Reference

K. R. Harinath et al., "Mind Watch: A Hybrid Framework for Mental Health Risk Prediction"
Department of Computer Science and Engineering, RGMCET, Nandyal, India.
