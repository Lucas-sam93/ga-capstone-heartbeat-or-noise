# Heart on Your Wrist
### Evaluating Consumer Wearables as Cardiac Screening Proxies Using Machine Learning

**Author:** Lucas Sam
**Program:** General Assembly Data Analytics Capstone Project
**Date:** March 2026

---

## Overview

Cardiovascular disease accounts for 30.5% of all deaths in Singapore — approximately 22 lives lost daily. Yet 49% of residents did not undergo a health checkup in the past twelve months. Consumer wearables are already worn by millions and continuously monitor heart rate and heart rate variability without requiring active participation. This project evaluates whether machine learning models trained on clinical ECG data can generalise to consumer wearable signals to detect cardiac rhythm irregularities, assessing their feasibility as low-barrier proxy screening tools.

**Research Question:** Can machine learning models trained on clinical cardiac data generalise to consumer wearable signals to detect heart rhythm irregularities, and can consumer wearables feasibly serve as proxy screening tools for the general population?

---

## Datasets

| Dataset | Role | Records | Source |
|---------|------|---------|--------|
| **Physionet 2017 AF Challenge** | Layer 1 training & evaluation | 8,187 single-lead ECG recordings (300 Hz) | [physionet.org](https://physionet.org/content/challenge-2017/1.0.0/) |
| **Personal Apple Watch SE** | Layer 2 N=1 case study | 478,077 heart rate records (Apr 2021 – Feb 2026) | Apple Health export |
| **MIMIC PERform AF** | Layer 2 primary validation | 35 subjects, 20 min finger PPG each (125 Hz) | [Zenodo](https://doi.org/10.5281/zenodo.6807402) |

**Physionet:** 279 noisy-class records excluded. Binary mapping: Normal (N → 0) vs Abnormal (AF + Other → 1). Final split: 61.6% Normal, 38.4% Abnormal. AF comprises 24% of Abnormal class.

**Apple Watch:** PPG optical sensor only (no ECG capability). Clinical anchor: June 18, 2025 ECG confirming cardiac irregularity. Role is exploratory — not used for training or primary validation.

**MIMIC PERform AF:** 19 AF + 16 NSR critically ill adults with simultaneous PPG and ECG. Pre-assigned clinical labels. Selected after UMass Simband access was unavailable before deadline.

---

## Methodology

### Feature Set (8 features, locked before modelling)

| Feature | Description |
|---------|-------------|
| RMSSD | Root mean square of successive RR interval differences |
| SDNN | Standard deviation of all RR intervals |
| Mean RR | Average inter-beat interval (ms) |
| pNN50 | Proportion of successive intervals differing by >50 ms |
| HR Mean | Average heart rate (bpm) |
| HR Std Dev | Heart rate variability |
| RR Skewness | Asymmetry of RR interval distribution |
| RR Kurtosis | Tail weight of RR interval distribution |

Frequency domain features (LF/HF ratio, HF power) permanently excluded — consumer wearables do not expose raw beat-by-beat intervals at consistent sampling rates.

### Evaluation Framework

- **Layer 1 (Clinical Benchmark):** Train and validate on Physionet ECG. Pre-registered criterion: Sensitivity ≥ 80% at Specificity ≥ 75%.
- **Layer 2 (Wearable Bridge):** Apply trained model to wearable PPG data. Assess signal transferability via three tiers: pipeline execution → signal detection → clinical concordance.

---

## Results

### Layer 1 — Model Selection

| Model | Sensitivity | Specificity | AUROC | Overfitting |
|---|---|---|---|---|
| Logistic Regression | 80.3% | 84.5% | 0.8810 | None |
| Random Forest | 80.0% | 86.7% | 0.9025 | +13.4% |
| XGBoost | 80.0% | 84.7% | 0.8874 | +13.6% |
| **SVM** | **80.0%** | **88.1%** | **0.8981** | **−1.2%** |

**Selected: SVM** — highest specificity, fewest false positives, no overfitting.

### Layer 1 — Held-Out Test (SVM, threshold 0.34)

| Metric | Value |
|---|---|
| Sensitivity | 84.4% |
| Specificity | 87.3% |
| AUROC | 0.9080 |
| AF Sensitivity | 98.8% |
| Pre-registered criterion | **PASS** |

### Layer 2 — Apple Watch N=1 (Null Result)

1,997 windows scored. No significant elevation around the clinical anchor event (all Mann-Whitney p > 0.7). **Tier 1 PASS, Tier 2 FAIL, Tier 3 FAIL.**

Three explanations: (1) signal modality gap — 5 of 8 features show large KS distances; (2) behavioural confound — reduced training intensity shifted HRV toward normal-appearing patterns; (3) the clinical ECG confirms an intraventricular conduction delay (waveform morphology), not a rhythm irregularity — HRV features cannot detect this type of abnormality.

### Layer 2 — MIMIC PERform AF Primary Validation

**At fixed ECG threshold (0.34):** 100% sensitivity, 12.5% specificity, AUROC 0.8586. All 8 features show large KS distances — the modality gap systematically inflates scores, pushing almost all subjects above threshold.

**After threshold recalibration (LOOCV, sensitivity-targeted):**

| Method | Sensitivity | Specificity |
|---|---|---|
| Fixed threshold (0.34) | 100% | 12.5% |
| Sensitivity-targeted LOOCV (0.8368) | 78.9% | 81.2% |
| Pre-registered criterion | ≥ 80% | ≥ 75% |

Bootstrap AUROC 95% CI: [0.7246, 0.9720]. Specificity failure is a calibration problem, not a discrimination failure.

---

## Conclusion

**The answer is a qualified yes.** The discriminative signal trained on clinical ECG transfers meaningfully to consumer wearable PPG (AUROC 0.86), but direct threshold generalisation fails. An ECG-calibrated threshold applied to PPG data over-flags — recalibration recovers clinically meaningful specificity.

The Apple Watch null result narrows the validated scope: HRV-based screening detects rhythm irregularities (beat-to-beat timing abnormalities) but cannot detect morphological abnormalities such as conduction delays, which affect waveform shape rather than inter-beat timing.

**For Singapore:** The screening gap is real — 49% unscreened, 22 CVD deaths daily. Consumer wearables already collect the data. The path forward requires PPG-specific threshold calibration, access to richer wearable signals (raw RR intervals), and validation on larger population-representative cohorts. The technology is not ready for autonomous screening today, but the discriminative foundation is present — the problem is calibration, not capability.

---

## BeatCheck — Screening App

A proof-of-concept web app applying the trained SVM to Apple Health exports.

- **Stack:** FastAPI + plain HTML/CSS/JS, deployed on Render
- **Input:** Apple Health CSV or XML export (client-side filtering to 90 days of heart rate data)
- **Output:** Percentage of 30-minute windows flagged, mapped to three risk tiers (Low < 10%, Intermediate 10–40%, High > 40%)
- **Threshold:** 0.8368 (LOOCV mean — domain-adapted for PPG)
- **Stress-tested:** Real 4.8-year Apple Health export → Intermediate tier, 38.2% flagged across 84 days

Not a medical device. Hard disclaimer required before upload.

---

## Limitations

- Not a clinical trial or validated medical device
- Apple Watch analysis is N=1 — findings cannot be generalised
- PPG measures cardiac activity through a fundamentally different mechanism than ECG
- Apple Watch SE lacks an electrical heart sensor; RMSSD is approximated from SDNN
- MIMIC validation uses 35 critically ill ICU subjects — differs from the general screening population
- Threshold recalibration and LOOCV computed on the same 35 subjects — indicative, not definitive
- HRV features detect rhythm irregularities only — not morphological abnormalities (conduction delays, ST changes)

---

## Project Structure

```
ga-capstone-heart-on-your-wrist/
├── app/                        # BeatCheck web app
│   ├── main.py                 # FastAPI backend
│   ├── pipeline.py             # ML inference pipeline
│   ├── models/                 # scaler.joblib + selected_model.joblib
│   └── static/index.html       # Frontend
├── data/                       # gitignored — not committed
│   ├── physionet/              # 8,528 ECG recordings
│   ├── apple_watch/            # Personal Apple Watch exports
│   ├── mimic_perform_af/       # MIMIC PERform AF CSVs
│   └── processed/              # Cleaned features & matrices
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_layer2_analysis.ipynb          # Apple Watch case study
│   ├── 05_mimic_perform_af_validation.ipynb
│   └── 06_technical_report.ipynb
├── src/
│   ├── features.py             # Physionet ECG feature extraction
│   ├── preprocess.py           # Apple Watch data cleaning
│   ├── evaluate.py             # Model evaluation functions
│   ├── apple_watch_features.py # Windowed Apple Watch features
│   └── mimic_perform_af_features.py
├── outputs/
│   ├── figures/                # All visualisations + figures_log.csv
│   ├── models/                 # gitignored — scaler, SVM, evaluation report
│   └── layer2/                 # KS results, scores, tier assessments
├── CLAUDE.md                   # Full methodological decision log
├── render.yaml                 # Deployment config
└── requirements.txt
```

`data/`, `data/processed/`, and `outputs/models/` are gitignored. Rerun notebooks to regenerate.

---

## Environment Setup

```bash
git clone https://github.com/LucasSam/ga-capstone-heart-on-your-wrist.git
conda create -n cvd_project python=3.13
conda activate cvd_project
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel -y
pip install wfdb xgboost joblib neurokit2
pip install -r app/requirements.txt
```

### Running BeatCheck Locally

```bash
cd app
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Future Work

- **PPG-specific training data:** Train or fine-tune models on PPG-labeled datasets to eliminate threshold recalibration dependency
- **Larger validation cohort:** Test on population-representative subjects rather than critically ill ICU patients
- **Raw RR interval access:** Partner with wearable manufacturers to access beat-by-beat intervals instead of summary statistics
- **Multi-device support:** Extend input parsing to Samsung Health, Fitbit, and Garmin exports
- **Regulatory pathway:** Explore IEC 62304 / FDA SaMD classification for clinical deployment
- **Longitudinal monitoring:** Track individual baselines over time rather than single-snapshot analysis

---

## Acknowledgements

Physionet 2017 AF Classification Challenge. MIMIC PERform AF dataset (Zenodo, doi.org/10.5281/zenodo.6807402). Personal health data collected via Apple Watch SE (1st Generation).
