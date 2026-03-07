"""
BeatCheck pipeline — data processing and inference.

Loads the SVM model and StandardScaler from app/models/ at import time.
Provides three public functions:
  - parse_apple_health_export(file_bytes) -> pd.DataFrame
  - parse_apple_health_xml(file_bytes) -> pd.DataFrame
  - process_and_predict(df) -> dict
"""

import io
import os
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ---------------------------------------------------------------------------
# Load model artefacts at import time
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_scaler = joblib.load(os.path.join(_MODEL_DIR, "scaler.joblib"))
_model = joblib.load(os.path.join(_MODEL_DIR, "selected_model.joblib"))

# Suppress sklearn warning about missing feature names (cosmetic only)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Feature column order — must match training exactly
_FEATURE_COLS = [
    "rmssd", "sdnn", "mean_rr", "pnn50",
    "hr_mean", "hr_std", "rr_skewness", "rr_kurtosis",
]

# Locked threshold — LOOCV mean from sensitivity-targeted LOOCV
_THRESHOLD = 0.8368


def parse_apple_health_export(file_bytes: bytes) -> pd.DataFrame:
    """Parse an Apple Health CSV export and return heart rate rows."""
    df = pd.read_csv(io.BytesIO(file_bytes))

    if "type" not in df.columns:
        raise ValueError(
            "Missing 'type' column — this does not appear to be an "
            "Apple Health CSV export."
        )

    df = df[df["type"] == "HKQuantityTypeIdentifierHeartRate"].copy()

    if df.empty:
        raise ValueError(
            "No heart rate records found in the uploaded file."
        )

    if "value" in df.columns:
        df = df.rename(columns={"value": "Value"})

    if "Value" not in df.columns:
        raise ValueError("Missing 'Value' column in heart rate data.")

    if "startDate" not in df.columns:
        raise ValueError("Missing 'startDate' column in heart rate data.")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["startDate"] = pd.to_datetime(df["startDate"], utc=False, errors="coerce")
    df = df.dropna(subset=["startDate", "Value"])

    return df[["startDate", "Value"]].reset_index(drop=True)


def _extract_xml_from_zip(file_bytes: bytes) -> bytes:
    """If file_bytes is a ZIP archive, extract export.xml from it."""
    buf = io.BytesIO(file_bytes)
    if not zipfile.is_zipfile(buf):
        return file_bytes

    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        # Apple Health exports contain apple_health_export/export.xml
        for name in zf.namelist():
            if name.lower().endswith("export.xml"):
                return zf.read(name)

        raise ValueError(
            "ZIP file does not contain an export.xml — "
            "please upload the Apple Health export ZIP or the "
            "export.xml file directly."
        )


def parse_apple_health_xml(file_bytes: bytes) -> pd.DataFrame:
    """Parse an Apple Health XML export (export.xml or Export.zip)."""
    # Handle ZIP archives (Apple Health exports as .zip)
    xml_bytes = _extract_xml_from_zip(file_bytes)

    # Strip DTD declaration — Apple Health XML includes <!DOCTYPE HealthData [...]>
    # which ET.iterparse cannot handle
    xml_str = xml_bytes.decode("utf-8", errors="replace")
    xml_str = re.sub(r'<!DOCTYPE\s+HealthData\s*\[.*?\]>', '', xml_str, flags=re.DOTALL)
    xml_bytes_clean = xml_str.encode("utf-8")

    # Use iterparse for memory efficiency on large exports
    records = []
    try:
        source = io.BytesIO(xml_bytes_clean)
        for event, elem in ET.iterparse(source, events=("end",)):
            if elem.tag == "Record" and elem.get("type") == "HKQuantityTypeIdentifierHeartRate":
                records.append({
                    "startDate": elem.get("startDate"),
                    "Value": elem.get("value"),
                })
            elem.clear()
    except ET.ParseError:
        raise ValueError(
            "Failed to parse XML — this does not appear to be a valid "
            "Apple Health XML export."
        )

    if not records:
        raise ValueError(
            "No heart rate records found in the uploaded XML file."
        )

    df = pd.DataFrame(records)
    df["startDate"] = pd.to_datetime(df["startDate"], utc=False, errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["startDate", "Value"])

    if df.empty:
        raise ValueError(
            "No valid heart rate records after parsing the XML file."
        )

    return df[["startDate", "Value"]].reset_index(drop=True)


def process_and_predict(df: pd.DataFrame) -> dict:
    """
    Build 30-min windows from heart rate data, extract features,
    run SVM inference, and return risk tier result.
    """
    # --- a. Filter to most recent 90 days ---
    df = df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"])
    max_date = df["startDate"].max()
    cutoff = max_date - pd.Timedelta(days=90)
    df = df[df["startDate"] >= cutoff]

    # --- b. Sort ascending ---
    df = df.sort_values("startDate").reset_index(drop=True)

    # --- c. Build 30-minute windows, 15-minute step ---
    window_size = pd.Timedelta(minutes=30)
    step_size = pd.Timedelta(minutes=15)
    start = df["startDate"].min()
    end = df["startDate"].max()

    features_list = []
    t = start
    while t + window_size <= end:
        w_end = t + window_size
        mask = (df["startDate"] >= t) & (df["startDate"] < w_end)
        window = df.loc[mask, "Value"].values

        t += step_size

        if len(window) < 10:
            continue

        hr = window.astype(float)
        rr = 60000.0 / hr  # RR intervals in ms

        rr_diff = np.diff(rr)

        features_list.append({
            "hr_mean": np.mean(hr),
            "hr_std": np.std(hr, ddof=1) if len(hr) > 1 else 0.0,
            "mean_rr": np.mean(rr),
            "rmssd": np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0,
            "sdnn": np.std(rr, ddof=1) if len(rr) > 1 else 0.0,
            "pnn50": (
                np.sum(np.abs(rr_diff) > 50) / len(rr_diff)
                if len(rr_diff) > 0 else 0.0
            ),
            "rr_skewness": skew(rr) if len(rr) > 2 else 0.0,
            "rr_kurtosis": kurtosis(rr) if len(rr) > 2 else 0.0,
        })

    # --- d. Build DataFrame and drop NaN windows ---
    if not features_list:
        raise ValueError(
            "Insufficient data — fewer than 5 valid windows found "
            "in the past 90 days."
        )

    feat_df = pd.DataFrame(features_list)[_FEATURE_COLS]
    feat_df = feat_df.dropna()

    # --- e. Minimum window check ---
    if len(feat_df) < 5:
        raise ValueError(
            "Insufficient data — fewer than 5 valid windows found "
            "in the past 90 days."
        )

    # --- f. Scale features (transform only — never refit) ---
    X_scaled = _scaler.transform(feat_df.values)

    # --- g. Predict probabilities ---
    probs = _model.predict_proba(X_scaled)[:, 1]

    # --- h. Apply threshold ---
    flagged = (probs >= _THRESHOLD).astype(int)

    # --- i. Percentage flagged ---
    total_windows = len(flagged)
    flagged_windows = int(flagged.sum())
    pct_flagged = round((flagged_windows / total_windows) * 100, 1)

    # --- j. Risk tier ---
    if pct_flagged < 10:
        risk_tier = "Low"
    elif pct_flagged < 40:
        risk_tier = "Intermediate"
    else:
        risk_tier = "High"

    # --- k. Days analysed ---
    days_analysed = int(df["startDate"].dt.date.nunique())

    return {
        "pct_flagged": pct_flagged,
        "total_windows": total_windows,
        "flagged_windows": flagged_windows,
        "risk_tier": risk_tier,
        "days_analysed": days_analysed,
    }
