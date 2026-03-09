"""
utils.py — Shared utility functions for the cardiac screening pipeline.

Extracts common operations used across features.py,
apple_watch_features.py, mimic_perform_af_features.py, and
the BeatCheck app pipeline to eliminate code duplication.
"""

import numpy as np
from scipy.stats import skew, kurtosis

from .constants import RR_MIN_MS, RR_MAX_MS


def filter_rr_intervals(rr_ms: np.ndarray) -> np.ndarray:
    """
    Remove physiologically implausible RR intervals.

    Retains only intervals within the 300-2000 ms range.
    Values outside this range indicate R-peak detection
    errors rather than genuine cardiac events.

    Parameters
    ----------
    rr_ms : np.ndarray
        RR intervals in milliseconds.

    Returns
    -------
    np.ndarray
        Filtered RR intervals within plausible bounds.
    """
    return rr_ms[(rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]


def compute_rr_features(rr_ms: np.ndarray) -> dict:
    """
    Compute the eight locked HRV features from RR intervals.

    Implements the exact same calculations used across Layer 1
    (Physionet ECG), Layer 2 (MIMIC PPG), and the BeatCheck app
    to ensure consistency.

    Parameters
    ----------
    rr_ms : np.ndarray
        RR intervals in milliseconds, already filtered for
        physiological plausibility.

    Returns
    -------
    dict
        Eight feature values keyed by their locked names:
        rmssd, sdnn, mean_rr, pnn50, hr_mean, hr_std,
        rr_skewness, rr_kurtosis.
    """
    successive_diffs = np.diff(rr_ms)

    # RMSSD — beat-to-beat variability magnitude
    rmssd = (
        np.sqrt(np.mean(successive_diffs ** 2))
        if len(successive_diffs) > 0 else 0.0
    )

    # SDNN — total HRV across recording window
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0.0

    # Mean RR — average inter-beat interval (ms)
    mean_rr = np.mean(rr_ms)

    # pNN50 — proportion of consecutive pairs differing >50ms
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (
        nn50 / len(successive_diffs)
        if len(successive_diffs) > 0 else 0.0
    )

    # HR Mean — average heart rate (bpm)
    hr_mean = 60000.0 / mean_rr

    # HR Std Dev — heart rate fluctuation across window
    hr_series = 60000.0 / rr_ms
    hr_std = np.std(hr_series, ddof=1) if len(rr_ms) > 1 else 0.0

    # RR Skewness — asymmetry of RR interval distribution
    rr_skewness = skew(rr_ms) if len(rr_ms) > 2 else 0.0

    # RR Kurtosis — tail weight of RR interval distribution
    rr_kurtosis = kurtosis(rr_ms) if len(rr_ms) > 2 else 0.0

    return {
        "rmssd":        rmssd,
        "sdnn":         sdnn,
        "mean_rr":      mean_rr,
        "pnn50":        pnn50,
        "hr_mean":      hr_mean,
        "hr_std":       hr_std,
        "rr_skewness":  rr_skewness,
        "rr_kurtosis":  rr_kurtosis,
    }
