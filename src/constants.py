"""
constants.py — Shared constants for the cardiac screening pipeline.

Centralises values used across multiple modules to eliminate
duplication and ensure consistency between Layer 1, Layer 2,
and the BeatCheck app.
"""

# ── Locked Feature Set ────────────────────────────────────────────
# Eight time-domain / statistical features derivable from both
# clinical ECG and wearable PPG. Locked before feature engineering
# began (February 2026). Order matters — must match training.
FEATURE_COLS = [
    "rmssd", "sdnn", "mean_rr", "pnn50",
    "hr_mean", "hr_std", "rr_skewness", "rr_kurtosis",
]

# ── Physiological Plausibility Bounds ─────────────────────────────
# RR interval range in milliseconds. Values outside these bounds
# indicate R-peak detection errors, not genuine cardiac events.
RR_MIN_MS = 300    # 200 bpm upper heart rate limit
RR_MAX_MS = 2000   # 30 bpm lower heart rate limit

# Minimum number of RR intervals required for reliable
# feature computation. Fewer intervals produce unreliable
# statistical features.
MIN_RR_COUNT = 10

# ── Console Output ────────────────────────────────────────────────
SEPARATOR = "=" * 55
