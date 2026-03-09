"""
features.py
===========
Feature extraction pipeline for Layer 1 clinical benchmark.

Extracts the eight locked HRV and RR interval features from
Physionet 2017 ECG recordings. All features are constrained
to measurements that consumer wearable devices can produce,
ensuring valid Layer 1 to Layer 2 generalisation.

Locked feature set (frequency domain excluded — Feb 2026):
    RMSSD, SDNN, Mean RR, pNN50,
    HR Mean, HR Std Dev, RR Skewness, RR Kurtosis

Called from: 02_feature_engineering.ipynb
"""

import os

import numpy as np
import pandas as pd
import wfdb
from wfdb import processing

from .constants import MIN_RR_COUNT, SEPARATOR
from .utils import filter_rr_intervals, compute_rr_features


def extract_features_single(record_path: str) -> dict | None:
    """
    Extract the eight locked HRV features from one ECG recording.

    Reads the raw ECG waveform, detects R-peaks, computes
    RR intervals, applies quality filtering, and returns
    the eight feature values as a dictionary.

    Returns None if the recording fails quality checks —
    the caller handles exclusion.

    Parameters
    ----------
    record_path : str
        Full path to the Physionet record WITHOUT file extension.
        Example: 'data/physionet/A00001'
        WFDB reads both A00001.mat and A00001.hea from this path.

    Returns
    -------
    dict or None
        Dictionary with eight feature keys and one 'valid' flag.
        Returns None if quality check fails.

        Keys: rmssd, sdnn, mean_rr, pnn50,
              hr_mean, hr_std, rr_skewness, rr_kurtosis, valid

    Raises
    ------
    Exception
        Any file read or processing error is caught internally
        and returns None to allow the pipeline to continue
        processing remaining recordings.
    """
    try:
        # ── Step 1: Read the raw ECG waveform ───────────────────
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # First lead only
        fs = record.fs                   # Sampling frequency (300 Hz)

        # ── Step 2: Detect R-peaks ───────────────────────────────
        xqrs = processing.XQRS(sig=signal, fs=fs)
        xqrs.detect(verbose=False)
        r_peaks = xqrs.qrs_inds

        # ── Step 3: Convert R-peak locations to RR intervals ─────
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / fs) * 1000.0

        # ── Step 4: Quality filtering ────────────────────────────
        rr_ms = filter_rr_intervals(rr_ms)

        if len(rr_ms) < MIN_RR_COUNT:
            return None

        # ── Step 5: Compute the eight locked features ─────────────
        features = compute_rr_features(rr_ms)
        features = {k: round(v, 4) for k, v in features.items()}
        features['valid'] = True
        return features

    except Exception:
        return None


def build_feature_matrix(physionet_dir: str, labels_path: str) -> pd.DataFrame:
    """
    Extract features from all Physionet recordings.

    Iterates over all records in the labels file, calls
    extract_features_single for each, collects results,
    and returns a clean dataframe ready for model training.

    Noisy class records (label '~') are excluded here.
    Failed extractions are logged and excluded.

    Parameters
    ----------
    physionet_dir : str
        Directory containing Physionet .mat and .hea files.
        Example: 'C:/Projects/GA Capstone Project/data/physionet'
    labels_path : str
        Path to REFERENCE-v3.csv containing record names
        and their clinical labels.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns:
        record, label, binary_label, rmssd, sdnn, mean_rr,
        pnn50, hr_mean, hr_std, rr_skewness, rr_kurtosis

        binary_label: 0 = Normal, 1 = Abnormal (AF + Other)

    Also prints
    -----------
    Progress every 500 records.
    Final extraction report showing success, failure,
    exclusion counts and class distribution.
    """
    # ── Load labels ──────────────────────────────────────────────
    labels_df = pd.read_csv(
        labels_path,
        header=None,
        names=['record', 'label']
    )

    # ── Exclude noisy class ──────────────────────────────────────
    # 279 records labelled '~' are excluded — these represent
    # signal quality failure, not rhythm classification.
    noisy_excluded = (labels_df['label'] == '~').sum()
    labels_df = labels_df[labels_df['label'] != '~'].copy()

    # ── Binary label mapping ─────────────────────────────────────
    # N  → 0 (Normal)
    # A  → 1 (Abnormal — Atrial Fibrillation)
    # O  → 1 (Abnormal — Other rhythm)
    labels_df['binary_label'] = labels_df['label'].map(
        {'N': 0, 'A': 1, 'O': 1}
    )

    # ── Extract features for each record ─────────────────────────
    results = []
    failed  = []

    total = len(labels_df)
    print(f"Starting feature extraction — {total} recordings")
    print(SEPARATOR)

    for i, row in labels_df.iterrows():
        record_path = os.path.join(
            physionet_dir, row['record']
        )
        features = extract_features_single(record_path)

        if features is not None:
            features['record']       = row['record']
            features['label']        = row['label']
            features['binary_label'] = row['binary_label']
            results.append(features)
        else:
            failed.append(row['record'])

        # Progress update every 500 records
        processed = len(results) + len(failed)
        if processed % 500 == 0:
            print(f"  Processed {processed:,} / {total:,} "
                  f"({processed/total*100:.1f}%)")

    # ── Build dataframe ──────────────────────────────────────────
    df = pd.DataFrame(results)

    # Reorder columns for clarity
    col_order = [
        'record', 'label', 'binary_label',
        'rmssd', 'sdnn', 'mean_rr', 'pnn50',
        'hr_mean', 'hr_std', 'rr_skewness', 'rr_kurtosis'
    ]
    df = df[col_order]

    # ── Print extraction report ──────────────────────────────────
    print("\n" + SEPARATOR)
    print("FEATURE EXTRACTION COMPLETE")
    print(SEPARATOR)
    print(f"Noisy records excluded:  {noisy_excluded:>6,}")
    print(f"Successful extractions:  {len(results):>6,}")
    print(f"Failed extractions:      {len(failed):>6,}")
    print(f"Final feature matrix:    {len(df):>6,} records")
    print(f"\nClass distribution:")
    print(f"  Normal   (0): "
          f"{(df['binary_label']==0).sum():,} "
          f"({(df['binary_label']==0).mean()*100:.1f}%)")
    print(f"  Abnormal (1): "
          f"{(df['binary_label']==1).sum():,} "
          f"({(df['binary_label']==1).mean()*100:.1f}%)")

    if failed:
        print(f"\nFailed records logged for review:")
        for r in failed[:10]:
            print(f"  {r}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    return df