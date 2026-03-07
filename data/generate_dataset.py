"""
STEP 2: Dataset Generator for MindWatch
========================================
Since real combined text+keystroke mental health datasets are rare,
this script creates a realistic synthetic dataset for academic use.

It uses real mental health text patterns based on research literature,
combined with statistically grounded keystroke behavior simulation.

Run: python data/generate_dataset.py
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Real-world style text samples
# ─────────────────────────────────────────────
AT_RISK_TEXTS = [
    "I can't stop thinking about how everything is falling apart around me",
    "I feel so empty inside, like nothing matters anymore",
    "Nobody would even notice if I just disappeared from everything",
    "I've been crying every night and I don't even know why",
    "I'm exhausted all the time but I can't sleep properly",
    "Everything feels hopeless and I don't see a way out of this",
    "I keep messing everything up no matter how hard I try",
    "I feel like a burden to everyone around me",
    "I can't concentrate on anything, my mind is always racing",
    "I've been isolating myself because I just can't face people",
    "I feel so anxious about everything, even small tasks feel overwhelming",
    "I don't enjoy anything anymore that I used to love",
    "I'm constantly worried something terrible is going to happen",
    "I feel disconnected from reality, like I'm watching my life from outside",
    "I can't stop thinking about all the ways I've failed in life",
    "My anxiety is so bad I can't leave the house some days",
    "I feel trapped with no way forward and no one who understands",
    "I've been having dark thoughts lately and I'm scared of myself",
    "I can't find any motivation to do even basic things anymore",
    "I feel like I'm drowning and no one can see me struggling",
    "The panic attacks are getting worse and I don't know what to do",
    "I've been feeling numb for weeks now, completely detached",
    "I hate myself and I don't know how to stop these thoughts",
    "I feel so alone even when I'm surrounded by people",
    "My mood swings are destroying my relationships and I can't control them",
    "I've been skipping meals because eating feels pointless",
    "I lay in bed all day because getting up feels impossible",
    "I keep having nightmares and waking up in a cold sweat",
    "I feel like I'm going crazy and no one takes me seriously",
    "Everything is too much right now and I just want to escape",
]

NOT_AT_RISK_TEXTS = [
    "Had a really productive day today, finished all my tasks ahead of schedule",
    "Went for a long walk this morning and felt so refreshed afterwards",
    "Caught up with old friends over dinner tonight, it was so great to reconnect",
    "Finally finished that book I've been reading, the ending was perfect",
    "Cooked a new recipe today and it turned out amazing",
    "Feeling really grateful for all the good things in my life right now",
    "Had a great workout session at the gym, feeling energized",
    "The weather was beautiful today so I spent time in the garden",
    "Got some really positive feedback on my project at work today",
    "Feeling optimistic about the future and excited about my plans",
    "Spent quality time with family this weekend, really recharged",
    "Made some good progress on my personal goals this week",
    "Tried a new coffee shop and discovered my new favorite drink",
    "Feeling confident and ready to tackle new challenges ahead",
    "Had a relaxing evening watching movies and eating popcorn",
    "My mindfulness practice is really helping me stay calm and focused",
    "Excited about the upcoming trip I've been planning for months",
    "Everything seems to be coming together nicely in my life",
    "Feeling strong and healthy after taking better care of myself",
    "Had a fun spontaneous adventure with friends today",
    "Learned something new today and it really sparked my curiosity",
    "Feeling content and at peace with where I am right now",
    "Celebrated a small win today and it really boosted my mood",
    "Getting back into hobbies I love has made such a difference",
    "The support from people around me means everything to me",
    "Starting to see real progress towards my long term goals",
    "Feeling motivated and inspired after a really good conversation",
    "A good night's sleep made all the difference to my day today",
    "Enjoying the simple pleasures of everyday life more and more",
    "Feeling balanced and in control of my emotions and reactions",
]


def simulate_keystroke_features(label: int) -> dict:
    """
    Simulate realistic keystroke dynamics based on mental health research.

    Research basis:
    - Epp et al. (2011): Stressed/anxious users type slower with more errors
    - Higher latency and error rates correlate with cognitive load
    - Depressed users show reduced motor speed

    Args:
        label: 1 = at risk, 0 = not at risk

    Returns:
        dict with typing_speed, keystroke_latency, error_rate
    """
    if label == 1:
        # At-risk: slower typing, longer pauses, more corrections
        typing_speed     = np.random.normal(loc=42, scale=8)    # keystrokes/sec
        keystroke_latency = np.random.normal(loc=0.22, scale=0.05)  # seconds
        error_rate        = np.random.normal(loc=0.10, scale=0.03)  # ratio
    else:
        # Healthy: faster, more confident typing
        typing_speed     = np.random.normal(loc=75, scale=10)
        keystroke_latency = np.random.normal(loc=0.09, scale=0.02)
        error_rate        = np.random.normal(loc=0.02, scale=0.01)

    # Clamp to realistic bounds
    typing_speed      = max(10.0, min(120.0, typing_speed))
    keystroke_latency = max(0.05, min(0.60,  keystroke_latency))
    error_rate        = max(0.00, min(0.30,  error_rate))

    return {
        "typing_speed":      round(typing_speed, 4),
        "keystroke_latency": round(keystroke_latency, 4),
        "error_rate":        round(error_rate, 4),
    }


def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate balanced mental health dataset with text + keystroke features.

    Args:
        n_samples: total number of samples (will be balanced 50/50)

    Returns:
        DataFrame with columns: text, label, typing_speed,
                                keystroke_latency, error_rate
    """
    records = []
    half = n_samples // 2

    # ── AT RISK samples (label = 1) ──
    for i in range(half):
        base_text = random.choice(AT_RISK_TEXTS)
        # Add slight variation so not all identical
        variations = [
            base_text,
            base_text + " I don't know what to do anymore.",
            base_text + " It's been weeks like this.",
            "Honestly, " + base_text.lower(),
        ]
        text = random.choice(variations)
        ks   = simulate_keystroke_features(label=1)
        records.append({"text": text, "label": 1, **ks})

    # ── NOT AT RISK samples (label = 0) ──
    for i in range(half):
        base_text = random.choice(NOT_AT_RISK_TEXTS)
        variations = [
            base_text,
            base_text + " Really happy about it.",
            base_text + " Life is good.",
            "Today, " + base_text.lower(),
        ]
        text = random.choice(variations)
        ks   = simulate_keystroke_features(label=0)
        records.append({"text": text, "label": 0, **ks})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("⏳ Generating dataset...")
    df = generate_dataset(n_samples=1000)

    output_path = "data/processed/dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Total samples : {len(df)}")
    print(f"   At-risk (1)   : {df['label'].sum()}")
    print(f"   Healthy (0)   : {(df['label'] == 0).sum()}")
    print(f"\nSample rows:")
    print(df.head(3).to_string())
