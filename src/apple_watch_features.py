"""
apple_watch_features.py — Layer 2 Apple Watch Feature Extraction

Extracts the locked 8-feature set from Apple Watch heart rate and HRV
data using sliding time-based windows. Features are designed to match
the Physionet feature matrix from Layer 1.

RMSSD is approximated as SDNN x 0.85. Apple Watch does not expose raw
beat-by-beat intervals required for direct RMSSD computation.

Calls: nothing from src/ — standalone module
Input: data/processed/heart_rate_clean.csv, data/processed/hrv_clean.csv
Output: returns DataFrames only — notebook handles saving
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ── Constants ─────────────────────────────────────────────────
RMSSD_APPROXIMATION_FACTOR = 0.85


def extract_hrv_features(hrv_path):
    """
    Extract SDNN and approximated RMSSD from Apple Watch HRV records.

    Each Apple Watch HRV record contains a single SDNN measurement (ms).
    RMSSD is approximated as SDNN x 0.85 because the watch does not
    expose raw beat-by-beat intervals.

    Parameters
    ----------
    hrv_path : str
        Absolute path to hrv_clean.csv.

    Returns
    -------
    pd.DataFrame
        Columns: startDate, sdnn, rmssd, anchor_period.
        One row per HRV record.
    """
    if not os.path.exists(hrv_path):
        raise FileNotFoundError(f"HRV file not found: {hrv_path}")

    df = pd.read_csv(hrv_path, parse_dates=['startDate'])
    df = df.sort_values('startDate').reset_index(drop=True)

    result = pd.DataFrame({
        'startDate':     df['startDate'],
        'sdnn':          df['value'],
        'rmssd':         df['value'] * RMSSD_APPROXIMATION_FACTOR,
        'anchor_period': df['anchor_period']
    })

    return result


def extract_hr_features(hr_path, window_size_minutes=30,
                        step_size_minutes=15, min_readings=10):
    """
    Extract HR-derived features using sliding time-based windows.

    Slides a fixed-width time window across the heart rate time series.
    For each window meeting the minimum readings threshold, computes
    six features from the HR values and derived RR intervals.

    RR intervals are derived as 60000 / HR (converting bpm to ms).
    pNN50 is computed from consecutive RR interval differences within
    each window.

    Parameters
    ----------
    hr_path : str
        Absolute path to heart_rate_clean.csv.
    window_size_minutes : int
        Width of each sliding window in minutes.
    step_size_minutes : int
        Step size between consecutive windows in minutes.
    min_readings : int
        Minimum number of HR readings required per window. Windows
        with fewer readings are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: window_start, window_end, hr_mean, hr_std, mean_rr,
        pnn50, rr_skewness, rr_kurtosis, n_readings, anchor_period.
        One row per valid window.

    Notes
    -----
    pNN50 is computed from RR intervals derived via 60000/HR, not from
    raw beat-by-beat timing. Averaged HR readings smooth out the
    beat-to-beat variation that pNN50 measures, introducing a known
    downward bias compared to Layer 1's ECG-derived pNN50. This is
    documented as a limitation and quantified in the KS gap analysis.
    """
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"HR file not found: {hr_path}")
    if window_size_minutes <= 0:
        raise ValueError("window_size_minutes must be positive")
    if step_size_minutes <= 0:
        raise ValueError("step_size_minutes must be positive")
    if min_readings < 2:
        raise ValueError("min_readings must be at least 2")

    df = pd.read_csv(hr_path, parse_dates=['startDate'], low_memory=False)
    df = df.sort_values('startDate').reset_index(drop=True)

    window_delta = pd.Timedelta(minutes=window_size_minutes)
    step_delta = pd.Timedelta(minutes=step_size_minutes)

    # Window grid: from first reading to last reading
    t_min = df['startDate'].iloc[0]
    t_max = df['startDate'].iloc[-1]

    windows = []
    window_start = t_min

    while window_start + window_delta <= t_max + step_delta:
        window_end = window_start + window_delta

        # Select readings within this window
        mask = (
            (df['startDate'] >= window_start)
            & (df['startDate'] < window_end)
        )
        chunk = df.loc[mask]

        if len(chunk) >= min_readings:
            hr_values = chunk['value'].values
            rr_values = 60000.0 / hr_values  # bpm to ms

            # Consecutive RR differences for pNN50
            rr_diffs = np.abs(np.diff(rr_values))
            pnn50 = (
                np.sum(rr_diffs > 50) / len(rr_diffs)
                if len(rr_diffs) > 0 else 0.0
            )

            # Determine anchor_period — majority vote within window
            anchor = chunk['anchor_period'].mode().iloc[0]

            windows.append({
                'window_start':  window_start,
                'window_end':    window_end,
                'hr_mean':       np.mean(hr_values),
                'hr_std':        np.std(hr_values, ddof=1),
                'mean_rr':       np.mean(rr_values),
                'pnn50':         pnn50,
                'rr_skewness':   skew(rr_values),
                'rr_kurtosis':   kurtosis(rr_values),
                'n_readings':    len(chunk),
                'anchor_period': anchor
            })

        window_start += step_delta

    result = pd.DataFrame(windows)

    return result


def build_apple_watch_feature_matrix(hr_path, hrv_path,
                                     window_size_minutes=30,
                                     step_size_minutes=15,
                                     min_readings=10):
    """
    Build the complete 8-feature Apple Watch matrix by merging HR
    windows with HRV records.

    Calls extract_hr_features() and extract_hrv_features(), then merges
    them using a vectorised sorted-interval join. An HR window is
    included only if at least one HRV record falls within the window
    time range. Windows with no matching HRV record are dropped (no
    imputation, no forward fill).

    When multiple HRV records fall within a single window, SDNN and
    RMSSD are averaged.

    Parameters
    ----------
    hr_path : str
        Absolute path to heart_rate_clean.csv.
    hrv_path : str
        Absolute path to hrv_clean.csv.
    window_size_minutes : int
        Width of each sliding window in minutes.
    step_size_minutes : int
        Step size between consecutive windows in minutes.
    min_readings : int
        Minimum number of HR readings required per window.

    Returns
    -------
    pd.DataFrame
        Columns: window_start, window_end, rmssd, sdnn, mean_rr, pnn50,
        hr_mean, hr_std, rr_skewness, rr_kurtosis, n_readings,
        anchor_period.
        Only windows with both sufficient HR readings and at least one
        HRV record are included.
    """
    # Extract features from each source
    hr_features = extract_hr_features(
        hr_path,
        window_size_minutes=window_size_minutes,
        step_size_minutes=step_size_minutes,
        min_readings=min_readings
    )

    hrv_features = extract_hrv_features(hrv_path)

    if hr_features.empty:
        raise ValueError("No valid HR windows produced — check data or "
                         "window parameters")

    # ── Vectorised interval join using searchsorted ───────────
    #
    # Instead of looping over every HR window and masking the HRV
    # DataFrame each time (O(W x H)), we use binary search to map
    # each HRV timestamp to its candidate windows in O(H x log W).
    # Because windows overlap (30min window, 15min step), each HRV
    # timestamp can fall in up to ceil(window/step) = 2 windows.
    # searchsorted finds the insertion point in the sorted window
    # starts, then we check offsets 1..max_overlap backwards to
    # find all containing windows — fully vectorised across all
    # HRV records with no Python row loop.

    w_starts = hr_features['window_start'].values
    w_ends = hr_features['window_end'].values
    hrv_times = hrv_features['startDate'].values
    n_windows = len(w_starts)

    # Maximum number of overlapping windows at any point in time
    max_overlap = int(np.ceil(window_size_minutes / step_size_minutes))

    # For each HRV time, find the first window_start > hrv_time
    insert_pos = np.searchsorted(w_starts, hrv_times, side='right')

    # Check candidate windows at offsets 1..max_overlap before
    # insert_pos — these are the only windows whose start <= hrv_time
    hrv_indices = np.arange(len(hrv_times))
    pair_chunks = []

    for offset in range(1, max_overlap + 1):
        candidate_w = insert_pos - offset
        # Filter: valid window index
        valid = (candidate_w >= 0) & (candidate_w < n_windows)
        valid_w = candidate_w[valid]
        valid_h = hrv_indices[valid]
        # Filter: HRV timestamp falls within window [start, end)
        in_window = hrv_times[valid] < w_ends[valid_w]
        if np.any(in_window):
            pair_chunks.append(
                np.column_stack([valid_w[in_window], valid_h[in_window]])
            )

    if not pair_chunks:
        raise ValueError("No windows survived merge — no HRV records "
                         "overlapped with any HR window")

    pairs = np.vstack(pair_chunks)

    # ── Aggregate HRV values per window ───────────────────────
    pair_df = pd.DataFrame(pairs, columns=['window_idx', 'hrv_idx'])
    pair_df['sdnn'] = hrv_features['sdnn'].values[pair_df['hrv_idx'].values]
    pair_df['rmssd'] = hrv_features['rmssd'].values[pair_df['hrv_idx'].values]

    hrv_agg = (
        pair_df
        .groupby('window_idx')
        .agg(sdnn=('sdnn', 'mean'), rmssd=('rmssd', 'mean'))
    )

    # ── Merge into final feature matrix ───────────────────────
    matched_windows = hr_features.iloc[hrv_agg.index].copy()
    matched_windows['sdnn'] = hrv_agg['sdnn'].values
    matched_windows['rmssd'] = hrv_agg['rmssd'].values

    # Reorder columns to match locked feature set
    feature_cols = [
        'window_start', 'window_end',
        'rmssd', 'sdnn', 'mean_rr', 'pnn50',
        'hr_mean', 'hr_std', 'rr_skewness', 'rr_kurtosis',
        'n_readings', 'anchor_period'
    ]
    result = matched_windows[feature_cols].reset_index(drop=True)

    # Round feature columns
    for col in ['rmssd', 'sdnn', 'mean_rr', 'pnn50',
                'hr_mean', 'hr_std', 'rr_skewness', 'rr_kurtosis']:
        result[col] = result[col].round(4)

    print(f"Apple Watch feature matrix built:")
    print(f"  HR windows produced:      {len(hr_features):,}")
    print(f"  HRV records available:    {len(hrv_features):,}")
    print(f"  Windows after merge:      {len(result):,}")
    print(f"  Windows dropped (no HRV): "
          f"{len(hr_features) - len(result):,}")

    return result
