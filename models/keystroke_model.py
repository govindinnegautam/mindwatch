"""
STEP 5: Keystroke Behavior Feature Extractor
=============================================
Implements Section 3.3 of the MindWatch paper.

Paper equations implemented here:
    S = N / t                                    (Typing Speed)
    L = (1/N-1) * Σ(k_{i+1} - k_i)            (Equation 3 - Keystroke Latency)
    E = Ne / N                                   (Error Rate)
    B = [S, L, E]                               (Behavioral vector)

Two modes:
    1. LIVE mode  : capture actual keystrokes in real-time (for demo)
    2. BATCH mode : process pre-recorded keystroke timestamps (for training)

Usage (batch):
    from models.keystroke_model import extract_features
    features = extract_features(timestamps=[0.0, 0.12, 0.25], corrections=1)

Usage (live):
    from models.keystroke_model import LiveKeystrokeCapture
    capture = LiveKeystrokeCapture()
    capture.start()
    # ... user types ...
    features = capture.get_features()
    capture.stop()
"""

import numpy as np
import time
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# BATCH MODE — process pre-recorded timestamps
# ─────────────────────────────────────────────────────────────

def extract_features(
    timestamps: List[float],
    corrections: int = 0
) -> np.ndarray:
    """
    Extract behavioral vector B = [speed, latency, error_rate]
    from keystroke timestamp log.

    Args:
        timestamps  : list of float timestamps (seconds) for each keypress
        corrections : number of correction keys pressed (backspace/delete)

    Returns:
        np.ndarray of shape [3] = [typing_speed, keystroke_latency, error_rate]

    Example:
        >>> timestamps = [0.0, 0.10, 0.21, 0.35, 0.50]
        >>> corrections = 1
        >>> B = extract_features(timestamps, corrections)
        >>> print(B)  # [10.0, 0.125, 0.2]
    """
    N = len(timestamps)

    if N < 2:
        print("  ⚠️  Too few keystrokes, returning zero vector")
        return np.zeros(3, dtype=np.float32)

    # ── Typing Speed S = N / total_time ────────────────────────────────────
    total_time = timestamps[-1] - timestamps[0]
    speed = N / total_time if total_time > 0 else 0.0

    # ── Keystroke Latency L (Equation 3 from paper) ─────────────────────────
    # Average time between consecutive key presses
    inter_key_delays = [
        timestamps[i + 1] - timestamps[i]
        for i in range(N - 1)
    ]
    latency = float(np.mean(inter_key_delays))

    # ── Error Rate E = corrections / total_keystrokes ────────────────────────
    error_rate = corrections / N

    # Clamp to realistic bounds (safety check)
    speed      = float(np.clip(speed,      0.0, 200.0))
    latency    = float(np.clip(latency,    0.0,   2.0))
    error_rate = float(np.clip(error_rate, 0.0,   1.0))

    return np.array([speed, latency, error_rate], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# LIVE MODE — real-time keystroke capture using pynput
# ─────────────────────────────────────────────────────────────

class LiveKeystrokeCapture:
    """
    Captures real keystroke dynamics from the user's keyboard.
    Uses pynput library to record timing without capturing actual keys.

    Privacy note: Only TIMING information is recorded, not key content.
    The actual characters typed are NOT stored.

    Usage:
        capture = LiveKeystrokeCapture()
        capture.start()
        input("Start typing in another window, then press Enter here...")
        capture.stop()
        features = capture.get_features()
        print(features)  # [speed, latency, error_rate]
    """

    def __init__(self):
        self._timestamps:  List[float] = []
        self._corrections: int         = 0
        self._listener                 = None
        self._running: bool            = False

    def _on_press(self, key) -> None:
        """Callback fired on each keypress — records timestamp only."""
        if not self._running:
            return

        self._timestamps.append(time.time())

        # Detect correction keys (backspace / delete)
        try:
            from pynput.keyboard import Key
            if key in (Key.backspace, Key.delete):
                self._corrections += 1
        except Exception:
            pass

    def start(self) -> None:
        """Start listening for keystrokes."""
        try:
            from pynput import keyboard
            self._timestamps  = []
            self._corrections = 0
            self._running     = True
            self._listener    = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
            print("🎹 Keystroke capture started (recording timing only, not content)")
        except ImportError:
            print("⚠️  pynput not installed. Run: pip install pynput")

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._listener:
            self._listener.stop()
        print(f"⏹  Capture stopped | {len(self._timestamps)} keystrokes recorded")

    def get_features(self) -> np.ndarray:
        """
        Extract features from captured session.

        Returns:
            np.ndarray [3] = [typing_speed, keystroke_latency, error_rate]
        """
        return extract_features(self._timestamps, self._corrections)

    def get_stats(self) -> dict:
        """Human-readable summary of captured session."""
        feats = self.get_features()
        return {
            "total_keystrokes":    len(self._timestamps),
            "correction_keys":     self._corrections,
            "typing_speed":        round(feats[0], 3),
            "keystroke_latency":   round(feats[1], 3),
            "error_rate":          round(feats[2], 3),
        }

    def reset(self) -> None:
        """Clear captured data for a new session."""
        self._timestamps  = []
        self._corrections = 0
        print("🔄 Capture buffer reset")


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Batch Mode Test ===")

    # Simulate at-risk user: slow typing with pauses
    at_risk_timestamps  = [0.0, 0.25, 0.55, 0.90, 1.30, 1.75, 2.25, 2.80]
    at_risk_corrections = 2
    at_risk_features    = extract_features(at_risk_timestamps, at_risk_corrections)
    print(f"\n  AT-RISK user:")
    print(f"    Speed  : {at_risk_features[0]:.2f} keys/sec")
    print(f"    Latency: {at_risk_features[1]:.3f} sec avg")
    print(f"    Errors : {at_risk_features[2]:.3f} ratio")

    # Simulate healthy user: fast confident typing
    healthy_timestamps  = [0.0, 0.08, 0.16, 0.25, 0.33, 0.41, 0.50, 0.58]
    healthy_corrections = 0
    healthy_features    = extract_features(healthy_timestamps, healthy_corrections)
    print(f"\n  HEALTHY user:")
    print(f"    Speed  : {healthy_features[0]:.2f} keys/sec")
    print(f"    Latency: {healthy_features[1]:.3f} sec avg")
    print(f"    Errors : {healthy_features[2]:.3f} ratio")
