import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats

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

# ── Constants ────────────────────────────────────────────────────

# Physiological plausibility bounds for RR intervals (ms).
# Matches src/features.py exactly.
RR_MIN_MS = 300    # 200 bpm upper limit
RR_MAX_MS = 2000   # 30 bpm lower limit

# Quality tier thresholds based on detected peak count.
PEAK_THRESHOLD_GREEN = 300
PEAK_THRESHOLD_AMBER = 100


def load_mimic_perform_af_records(af_dir, non_af_dir):
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


def clean_ppg_signal(ppg_series):
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


def extract_rr_intervals(ppg_signal, sampling_rate=125):
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
        rr_ms = rr_ms[
            (rr_ms >= RR_MIN_MS) &
            (rr_ms <= RR_MAX_MS)
        ]

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


def compute_mimic_features(rr_intervals, subject_id, label,
                           quality_tier='green'):
    """
    Compute the eight locked HRV features from RR intervals.

    Feature definitions match src/features.py exactly to ensure
    valid comparison between Layer 1 (clinical ECG) and Layer 2
    (wearable PPG) results.

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

    # Successive differences (used by RMSSD and pNN50)
    successive_diffs = np.diff(rr_intervals)

    # RMSSD — root mean square of successive RR differences
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    # SDNN — standard deviation of all RR intervals
    sdnn = np.std(rr_intervals, ddof=1)

    # Mean RR — average inter-beat interval in milliseconds
    mean_rr = np.mean(rr_intervals)

    # pNN50 — proportion of consecutive pairs differing >50ms
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (nn50 / len(successive_diffs)
             if len(successive_diffs) > 0 else 0.0)

    # HR Mean — average heart rate from mean RR
    hr_mean = 60000.0 / mean_rr

    # HR Std Dev — variability of instantaneous heart rate
    hr_series = 60000.0 / rr_intervals
    hr_std = np.std(hr_series, ddof=1)

    # RR Skewness — asymmetry of RR distribution
    rr_skewness = stats.skew(rr_intervals)

    # RR Kurtosis — tail weight of RR distribution
    rr_kurtosis = stats.kurtosis(rr_intervals)

    return {
        'subject_id':   subject_id,
        'label':        label,
        'quality_tier': quality_tier,
        'rmssd':        round(rmssd, 4),
        'sdnn':         round(sdnn, 4),
        'mean_rr':      round(mean_rr, 4),
        'pnn50':        round(pnn50, 4),
        'hr_mean':      round(hr_mean, 4),
        'hr_std':       round(hr_std, 4),
        'rr_skewness':  round(rr_skewness, 4),
        'rr_kurtosis':  round(rr_kurtosis, 4)
    }


def build_mimic_feature_matrix(af_dir, non_af_dir):
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
    print("\n" + "=" * 55)
    print("MIMIC PERform AF — FEATURE EXTRACTION COMPLETE")
    print("=" * 55)
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
