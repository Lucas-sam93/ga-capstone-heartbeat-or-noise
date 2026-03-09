"""
preprocess.py
=============
Apple Watch data cleaning functions for Layer 2 analysis.
All functions accept raw extracted CSV data and return
cleaned, validated DataFrames ready for feature engineering.

Called from: 02_feature_engineering.ipynb
"""

import os

import numpy as np
import pandas as pd

# Physiological plausibility thresholds
# Based on published normal ranges for adult humans
THRESHOLDS = {
    'hrv':              {'min': 5,   'max': 200, 'unit': 'ms'},
    'heart_rate':       {'min': 30,  'max': 220, 'unit': 'count/min'},
    'resting_hr':       {'min': 30,  'max': 120, 'unit': 'count/min'},
    'walking_hr':       {'min': 50,  'max': 180, 'unit': 'count/min'},
    'respiratory_rate': {'min': 4,   'max': 40,  'unit': 'count/min'},
}

# Clinical anchor date — confirmed ECG event
ANCHOR_DATE = pd.Timestamp('2025-06-18')

# Anchor period window definitions in days
ANCHOR_WINDOWS = {
    'baseline':    (-9999, -91),
    'pre_anchor':  (-90,   -1),
    'post_anchor': (0,      90),
    'follow_up':   (91,   9999),
}


def clean_metric(metric_name: str, raw_dir: str, proc_dir: str) -> dict:
    """
    Clean a single Apple Watch metric from raw CSV.

    Applies timestamp parsing, numeric conversion,
    physiological outlier removal, temporal feature
    derivation, and clinical anchor period labelling.

    Parameters
    ----------
    metric_name : str
        Metric name matching raw CSV filename prefix.
        Must be one of: hrv, heart_rate, resting_hr,
        walking_hr, respiratory_rate
    raw_dir : str
        Directory containing raw CSV files from extraction
    proc_dir : str
        Directory to save cleaned output CSV files

    Returns
    -------
    dict
        Cleaning report containing:
        - metric: str
        - initial_records: int
        - non_numeric: int
        - outliers_removed: int
        - final_records: int
        - retention_pct: float

    Raises
    ------
    FileNotFoundError
        If raw CSV file does not exist at expected path
    ValueError
        If metric_name is not in THRESHOLDS dictionary
    """
    filepath = os.path.join(raw_dir, f'{metric_name}_raw.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw file not found: {filepath}\n"
            f"Run extraction pipeline in "
            f"01_data_exploration.ipynb first."
        )

    if metric_name not in THRESHOLDS:
        raise ValueError(
            f"Unknown metric: {metric_name}\n"
            f"Valid options: {list(THRESHOLDS.keys())}"
        )

    df = pd.read_csv(filepath)
    initial_count = len(df)

    # ── Step 1: Parse timestamps ─────────────────────────────────
    # Preserve +0800 Singapore timezone — no UTC conversion
    df['startDate'] = pd.to_datetime(df['startDate'])

    # ── Step 2: Convert values to numeric ────────────────────────
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    non_numeric = int(df['value'].isna().sum())
    df = df.dropna(subset=['value'])

    # ── Step 3: Apply physiological thresholds ───────────────────
    thresh = THRESHOLDS[metric_name]
    mask = (
        (df['value'] < thresh['min']) |
        (df['value'] > thresh['max'])
    )
    outliers_removed = int(mask.sum())
    df = df[~mask]

    # ── Step 4: Sort chronologically ────────────────────────────
    df = df.sort_values('startDate').reset_index(drop=True)

    # ── Step 5: Derive temporal features ─────────────────────────
    df['date']  = df['startDate'].dt.date
    df['year']  = df['startDate'].dt.year
    df['month'] = df['startDate'].dt.month
    df['hour']  = df['startDate'].dt.hour
    df['dow']   = df['startDate'].dt.day_name()

    # ── Step 6: Apply clinical anchor period labels ───────────────
    # Anchor date: June 18 2025 — confirmed ECG event
    df['days_from_anchor'] = (
        pd.to_datetime(df['date']) - ANCHOR_DATE
    ).dt.days

    # Derive bins and labels from ANCHOR_WINDOWS to avoid duplication
    period_names = list(ANCHOR_WINDOWS.keys())
    bins = [ANCHOR_WINDOWS[period_names[0]][0]]  # leftmost edge
    for name in period_names:
        bins.append(ANCHOR_WINDOWS[name][1])

    df['anchor_period'] = pd.cut(
        df['days_from_anchor'],
        bins=bins,
        labels=period_names
    )

    # ── Step 7: Save cleaned file ─────────────────────────────────
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(
        proc_dir, f'{metric_name}_clean.csv'
    )
    df.to_csv(out_path, index=False)

    return {
        'metric':           metric_name,
        'initial_records':  initial_count,
        'non_numeric':      non_numeric,
        'outliers_removed': outliers_removed,
        'final_records':    len(df),
        'retention_pct':    round(
            len(df) / initial_count * 100, 1
        )
    }


def run_cleaning_pipeline(raw_dir: str, proc_dir: str) -> list[dict]:
    """
    Run cleaning pipeline across all Apple Watch metrics.

    Calls clean_metric for each metric in the confirmed
    final dataset. Prints and returns a cleaning report.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw extracted CSV files
    proc_dir : str
        Directory to save cleaned output CSV files

    Returns
    -------
    list of dict
        One cleaning report dictionary per metric
    """
    metrics = [
        'hrv',
        'heart_rate',
        'resting_hr',
        'walking_hr',
        'respiratory_rate',
    ]

    reports = []
    for metric in metrics:
        report = clean_metric(metric, raw_dir, proc_dir)
        reports.append(report)

    return reports
