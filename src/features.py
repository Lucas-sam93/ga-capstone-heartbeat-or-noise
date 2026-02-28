import os
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from scipy import stats

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

# ── Constants ────────────────────────────────────────────────────

# Minimum number of RR intervals required for reliable
# feature computation. Recordings with fewer intervals
# are flagged and excluded rather than producing
# unreliable feature values.
MIN_RR_COUNT = 10

# Physiological plausibility bounds for RR intervals
# in milliseconds. Values outside these bounds indicate
# R-peak detection errors, not genuine cardiac events.
RR_MIN_MS = 300    # 200 bpm upper limit
RR_MAX_MS = 2000   # 30 bpm lower limit


def extract_features_single(record_path):
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
        # wfdb reads the .mat signal file and .hea header file.
        # The header contains sampling frequency and lead info.
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # First lead only
        fs = record.fs                   # Sampling frequency (300 Hz)

        # ── Step 2: Detect R-peaks ───────────────────────────────
        # XQRS is wfdb's built-in R-peak detector.
        # It finds the sharp spikes in the ECG waveform that
        # correspond to each ventricular contraction.
        # Output is an array of sample indices where R-peaks occur.
        xqrs = processing.XQRS(sig=signal, fs=fs)
        xqrs.detect(verbose=False)
        r_peaks = xqrs.qrs_inds

        # ── Step 3: Convert R-peak locations to RR intervals ─────
        # Subtract consecutive R-peak sample indices to get
        # the number of samples between beats, then convert
        # to milliseconds using the sampling frequency.
        # Formula: (samples between peaks / samples per second)
        #          * 1000 = milliseconds
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / fs) * 1000.0

        # ── Step 4: Quality filtering ────────────────────────────
        # Remove physiologically implausible RR intervals.
        # Values outside 300-2000ms indicate detection errors.
        rr_ms = rr_ms[
            (rr_ms >= RR_MIN_MS) &
            (rr_ms <= RR_MAX_MS)
        ]

        # Require minimum number of valid intervals.
        # Fewer than MIN_RR_COUNT intervals cannot produce
        # reliable statistical features.
        if len(rr_ms) < MIN_RR_COUNT:
            return None

        # ── Step 5: Compute the eight locked features ─────────────

        # RMSSD
        # Root mean square of successive RR differences.
        # Measures beat-to-beat variability — how much the
        # interval changes from one beat to the next.
        successive_diffs = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))

        # SDNN
        # Standard deviation of all RR intervals.
        # Captures total variability across the full window.
        sdnn = np.std(rr_ms, ddof=1)

        # Mean RR
        # Average inter-beat interval in milliseconds.
        mean_rr = np.mean(rr_ms)

        # pNN50
        # Proportion of consecutive pairs differing by >50ms.
        # Higher values indicate stronger parasympathetic tone.
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = nn50 / len(successive_diffs) if len(
            successive_diffs) > 0 else 0.0

        # HR Mean
        # Average heart rate derived from mean RR interval.
        # Formula: 60,000ms per minute / mean RR in ms
        hr_mean = 60000.0 / mean_rr

        # HR Standard Deviation
        # Variability of instantaneous heart rate values.
        # Each RR interval converted to its equivalent HR.
        hr_series = 60000.0 / rr_ms
        hr_std = np.std(hr_series, ddof=1)

        # RR Skewness
        # Asymmetry of the RR interval distribution.
        # Normal sinus rhythm produces near-zero skewness.
        # Arrhythmias often produce asymmetric distributions.
        rr_skewness = stats.skew(rr_ms)

        # RR Kurtosis
        # Tail weight of the RR interval distribution.
        # Captures how concentrated or spread the intervals are.
        rr_kurtosis = stats.kurtosis(rr_ms)

        return {
            'rmssd':       round(rmssd, 4),
            'sdnn':        round(sdnn, 4),
            'mean_rr':     round(mean_rr, 4),
            'pnn50':       round(pnn50, 4),
            'hr_mean':     round(hr_mean, 4),
            'hr_std':      round(hr_std, 4),
            'rr_skewness': round(rr_skewness, 4),
            'rr_kurtosis': round(rr_kurtosis, 4),
            'valid':       True
        }

    except Exception:
        # Any read or processing failure returns None.
        # The pipeline logs these as failed extractions.
        return None


def build_feature_matrix(physionet_dir, labels_path):
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
    print("="*55)

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
    print("\n" + "="*55)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*55)
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