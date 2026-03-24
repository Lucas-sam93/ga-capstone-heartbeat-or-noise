"""
mimic_perform_af_features.py
============================
Feature extraction pipeline for Layer 2 primary validation
using MIMIC PERform AF dataset (Zenodo).

Extracts the eight locked HRV features from finger PPG recordings
via NeuroKit2 peak detection. Features match src/features.py exactly
to ensure valid Layer 1 to Layer 2 comparison.

Locked feature set (frequency domain excluded):
    RMSSD, SDNN, Mean RR, pNN50,
    HR Mean, HR Std Dev, RR Skewness, RR Kurtosis

Called from: 05_mimic_perform_af_validation.ipynb
"""

import os

import numpy as np
import pandas as pd
import neurokit2 as nk

from .constants import SEPARATOR
from .utils import filter_rr_intervals, compute_rr_features

# Quality tier thresholds based on detected peak count.
PEAK_THRESHOLD_GREEN = 300
PEAK_THRESHOLD_AMBER = 100


def load_mimic_perform_af_records(af_dir: str, non_af_dir: str) -> list[dict]:
    """
    Load all MIMIC PERform AF CSV files from AF and non-AF directories.

    Each CSV file contains columns: Time, PPG, ECG, resp (some files
    lack resp). Only the PPG column is extracted for downstream
    peak detection.

    Parameters
    ----------
    af_dir : str
        Path to directory containing AF subject CSV files.
        These subjects are labelled 1 (Abnormal).
    non_af_dir : str
        Path to directory containing non-AF subject CSV files.
        These subjects are labelled 0 (Normal).

    Returns
    -------
    list of dict
        Each dict contains:
        - subject_id : str — filename without '_data.csv' suffix
        - ppg_signal : pd.Series — raw PPG values from the CSV
        - label : int — 1 for AF, 0 for non-AF

    Raises
    ------
    ValueError
        If either directory contains no CSV data files.
    """
    records = []

    for directory, label in [(af_dir, 1), (non_af_dir, 0)]:
        csv_files = sorted([
            f for f in os.listdir(directory)
            if f.endswith('_data.csv')
        ])

        if len(csv_files) == 0:
            raise ValueError(
                f"No CSV data files found in {directory}"
            )

        for filename in csv_files:
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            subject_id = filename.replace('_data.csv', '')

            records.append({
                'subject_id': subject_id,
                'ppg_signal': df['PPG'],
                'label': label
            })

    return records


def clean_ppg_signal(ppg_series: pd.Series) -> pd.Series:
    """
    Clean a PPG signal by interpolating null values.

    Applies linear interpolation to fill interior nulls, then
    forward-fills and back-fills any remaining edge nulls that
    interpolation cannot reach.

    Parameters
    ----------
    ppg_series : pd.Series
        Raw PPG signal values. May contain NaN entries.

    Returns
    -------
    pd.Series
        Cleaned PPG signal with no null values.
    """
    cleaned = ppg_series.interpolate(method='linear')
    cleaned = cleaned.ffill().bfill()
    return cleaned


def extract_rr_intervals(
        ppg_signal: pd.Series | np.ndarray,
        sampling_rate: int = 125) -> tuple[np.ndarray | None, int, str]:
    """
    Detect PPG peaks and compute RR intervals in milliseconds.

    Uses NeuroKit2 ppg_peaks() for peak detection on the cleaned
    PPG signal. Converts peak indices to RR intervals in ms,
    then applies physiological plausibility filtering (300-2000 ms).

    Assigns a quality tier based on detected peak count:
        - green:  >= 300 peaks (high confidence)
        - amber:  100-299 peaks (usable with flag)
        - red:    < 100 peaks (insufficient — skip subject)

    Parameters
    ----------
    ppg_signal : pd.Series or np.ndarray
        Cleaned PPG signal (no nulls).
    sampling_rate : int, optional
        Sampling frequency in Hz. Default 125 (MIMIC PERform AF).

    Returns
    -------
    tuple of (np.ndarray or None, int, str)
        - rr_intervals : RR intervals in milliseconds after
          physiological filtering, or None if peak detection fails
        - peak_count : number of peaks detected (0 if failed)
        - quality_tier : 'green', 'amber', or 'red'
    """
    try:
        _, info = nk.ppg_peaks(
            ppg_signal,
            sampling_rate=sampling_rate
        )
        peak_indices = info['PPG_Peaks']
        peak_count = len(peak_indices)

        if peak_count < 2:
            return None, peak_count, 'red'

        # Convert peak indices to RR intervals in milliseconds
        rr_samples = np.diff(peak_indices)
        rr_ms = (rr_samples / sampling_rate) * 1000.0

        # Apply physiological plausibility filtering
        rr_ms = filter_rr_intervals(rr_ms)

        # Assign quality tier
        if peak_count >= PEAK_THRESHOLD_GREEN:
            quality_tier = 'green'
        elif peak_count >= PEAK_THRESHOLD_AMBER:
            quality_tier = 'amber'
        else:
            quality_tier = 'red'

        return rr_ms, peak_count, quality_tier

    except Exception:
        return None, 0, 'red'


def compute_mimic_features(
        rr_intervals: np.ndarray, subject_id: str,
        label: int, quality_tier: str = 'green') -> dict:
    """
    Compute the eight locked HRV features from RR intervals.

    Delegates to the shared compute_rr_features() to ensure
    feature definitions match Layer 1 exactly.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds after physiological filtering.
    subject_id : str
        Subject identifier from the filename.
    label : int
        Ground truth label (1 = AF/Abnormal, 0 = NSR/Normal).
    quality_tier : str, optional
        Peak detection quality tier ('green' or 'amber').
        Default 'green'.

    Returns
    -------
    dict
        Dictionary with keys: subject_id, label, quality_tier,
        rmssd, sdnn, mean_rr, pnn50, hr_mean, hr_std,
        rr_skewness, rr_kurtosis

    Raises
    ------
    ValueError
        If rr_intervals is None or contains fewer than 100 values.
    """
    if rr_intervals is None or len(rr_intervals) < 100:
        raise ValueError(
            f"Subject {subject_id}: rr_intervals is None or has "
            f"fewer than 100 values "
            f"(got {0 if rr_intervals is None else len(rr_intervals)})"
        )

    features = compute_rr_features(rr_intervals)
    features = {k: round(v, 4) for k, v in features.items()}
    features['subject_id'] = subject_id
    features['label'] = label
    features['quality_tier'] = quality_tier
    return features


def build_mimic_feature_matrix(
        af_dir: str, non_af_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Orchestrate full feature extraction pipeline for all 35 subjects.

    For each subject: loads PPG signal, cleans nulls, detects peaks,
    extracts RR intervals, computes features. Subjects with red
    quality tier (< 100 peaks) are skipped and logged.

    Parameters
    ----------
    af_dir : str
        Path to directory containing AF subject CSV files.
    non_af_dir : str
        Path to directory containing non-AF subject CSV files.

    Returns
    -------
    tuple of (pd.DataFrame, list)
        - features_df : DataFrame with columns subject_id, label,
          quality_tier, rmssd, sdnn, mean_rr, pnn50, hr_mean,
          hr_std, rr_skewness, rr_kurtosis
        - failed_subjects : list of subject_id strings that were
          skipped due to red quality tier or insufficient RR intervals
    """
    records = load_mimic_perform_af_records(af_dir, non_af_dir)

    results = []
    failed_subjects = []
    green_count = 0
    amber_count = 0
    red_count = 0

    for record in records:
        subject_id = record['subject_id']
        label = record['label']

        # Step 1: Clean PPG signal
        ppg_clean = clean_ppg_signal(record['ppg_signal'])

        # Step 2: Extract RR intervals via peak detection
        rr_intervals, peak_count, quality_tier = extract_rr_intervals(
            ppg_clean
        )

        # Step 3: Handle by quality tier
        if quality_tier == 'red':
            red_count += 1
            failed_subjects.append(subject_id)
            print(f"  SKIP  {subject_id} — red tier "
                  f"({peak_count} peaks)")
            continue

        # Check if enough RR intervals survive filtering
        if rr_intervals is None or len(rr_intervals) < 100:
            red_count += 1
            failed_subjects.append(subject_id)
            rr_count = 0 if rr_intervals is None else len(rr_intervals)
            print(f"  SKIP  {subject_id} — insufficient RR intervals "
                  f"after filtering ({rr_count})")
            continue

        if quality_tier == 'amber':
            amber_count += 1
            print(f"  FLAG  {subject_id} — amber tier "
                  f"({peak_count} peaks)")
        else:
            green_count += 1

        # Step 4: Compute features
        features = compute_mimic_features(
            rr_intervals, subject_id, label, quality_tier
        )
        results.append(features)

    # Build DataFrame
    features_df = pd.DataFrame(results)

    # Print summary
    total = len(records)
    print("\n" + SEPARATOR)
    print("MIMIC PERform AF — FEATURE EXTRACTION COMPLETE")
    print(SEPARATOR)
    print(f"Total subjects attempted:  {total}")
    print(f"  Green (>= 300 peaks):    {green_count}")
    print(f"  Amber (100-299 peaks):   {amber_count}")
    print(f"  Red / skipped:           {red_count}")
    print(f"Final feature matrix:      {len(features_df)} subjects")

    if len(features_df) > 0:
        af_count = (features_df['label'] == 1).sum()
        nsr_count = (features_df['label'] == 0).sum()
        print(f"\nClass distribution:")
        print(f"  AF (Abnormal=1):  {af_count}")
        print(f"  NSR (Normal=0):   {nsr_count}")

    return features_df, failed_subjects
