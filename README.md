# Heartbeat or Noise?
### Evaluating Consumer Wearables as Cardiac Screening Proxies Using Machine Learning

**Author:** Lucas Sam
**Program:** General Assembly Data Analytics Capstone Project
**Date:** March 2026

---

## Overview

Cardiovascular disease accounts for 30.5% of all deaths in Singapore, equating to approximately 22 lives lost daily. Despite national screening efforts, approximately 49% of Singapore residents did not undergo a health checkup in the past twelve months, leaving a significant proportion of the population unaware of underlying cardiac risk.

Consumer wearables — including smartwatches from Apple and Samsung — are worn by millions daily and continuously monitor physiological signals including heart rate and heart rate variability. This project evaluates whether machine learning models trained on clinical ECG data can generalise to consumer wearable signals to detect cardiac rhythm irregularities, assessing their feasibility as low-barrier proxy screening tools that could prompt at-risk individuals toward formal clinical evaluation.

---

## Research Question

Can machine learning models trained on clinical cardiac data generalise to consumer wearable signals to detect heart rhythm irregularities, and can consumer wearables feasibly serve as proxy screening tools to identify individuals who would benefit from formal cardiac evaluation?

---

## Project Contribution

While Apple's AF detection algorithm represents a clinically validated application of consumer wearable cardiac monitoring, its scope is limited to atrial fibrillation specifically and its methodology remains proprietary and non-reproducible. This project extends beyond AF detection to classify a broader range of cardiac rhythm irregularities — normal versus any abnormal rhythm warranting clinical attention — using a transparent, reproducible machine learning methodology built entirely on publicly available clinical data. In doing so, it provides an open, auditable framework for evaluating whether consumer wearables can serve as a low-barrier proxy screening tool capable of prompting at-risk individuals toward formal clinical evaluation before an acute cardiac event occurs.

---

## Datasets

### Primary — Physionet 2017 AF Classification Challenge
- 8,528 single-lead ECG recordings sampled at 300 Hz
- 8,249 usable recordings after excluding 279 noisy-class (`~`) records
- Binary label mapping: Normal (N → 0) versus Abnormal (AF + Other → 1)
- Class distribution after feature extraction: 5,042 Normal (61.6%), 3,145 Abnormal (38.4%)
- AF within Abnormal class: 754 records (24.0%)
- Final feature matrix: 8,187 records (62 failed quality filtering — 0.75% failure rate)
- Source: https://physionet.org/content/challenge-2017/1.0.0/

### Secondary — Personal Apple Watch Data (N=1 Case Study)
- **Device:** Apple Watch SE 1st Generation — PPG optical sensor only. No electrical heart sensor.
- **Temporal coverage:** April 2021 — February 2026 (4.8 years)
- **Clinical anchor point:** ECG conducted June 18, 2025 confirming cardiac irregularity
- **Role:** Exploratory N=1 case study appendix — not used for model training or primary validation

| Metric | Records | Status |
|--------|--------:|--------|
| Heart rate (continuous) | 478,077 | Cleaned (1 artifact removed) |
| Respiratory rate | 20,372 | Complete |
| Sleep analysis | 7,967 | Complete |
| Heart rate variability (SDNN) | 5,456 | Cleaned (29 non-numeric removed) |
| Workouts | 1,982 | Complete |
| Walking heart rate average | 1,557 | Complete |
| Resting heart rate | 1,491 | Complete |

### Layer 2 Primary — UMass Simband (UMMC-DB) [Pending]
- **Device:** Samsung Simband wrist-worn smartwatch — PPG optical sensor, 128 Hz
- **Ground truth:** Simultaneous Holter ECG, cardiologist-labelled 30-second segments
- **Labels:** NSR (0), AF (1), PAC/PVC (2), Noisy (4)
- **Subjects:** 37–41 ambulatory clinic patients, approximately 10 with AF
- **Source:** Synapse syn25005551 (registration required)
- **Status:** Dataset identified, Synapse access not yet obtained

---

## Methodology

### Layer 1 — Clinical Benchmark

A binary cardiac rhythm classifier is trained and validated on the Physionet dataset using only features extractable from consumer wearable devices. This deliberate constraint ensures the model reflects real-world deployment conditions. Sensitivity to abnormal rhythms is the primary evaluation metric, consistent with the screening context in which missing a true positive carries greater consequence than generating a false alarm.

**Minimum performance threshold (pre-registered):** Sensitivity ≥ 0.80 at Specificity ≥ 0.75

**Locked feature set** (8 features — frequency domain permanently excluded):

| Feature | Description |
|---------|-------------|
| RMSSD | Root mean square of successive RR interval differences |
| SDNN | Standard deviation of all RR intervals |
| Mean RR | Average inter-beat interval (ms) |
| pNN50 | Proportion of successive intervals differing by >50ms |
| HR Mean | Average heart rate derived from RR intervals |
| HR Std Dev | Variability of instantaneous heart rate |
| RR Skewness | Asymmetry of the RR interval distribution |
| RR Kurtosis | Tail weight of the RR interval distribution |

Frequency domain features (LF/HF ratio, HF power) were permanently excluded because consumer wearables do not expose raw beat-by-beat interval sequences at consistent sampling rates.

### Layer 2 — Consumer Wearable Bridge

The trained model is applied to wearable data to evaluate signal transferability from clinical ECG to consumer PPG-derived measurements. Success is assessed across three tiers:

- **Tier 1 — Foundational Applicability:** The feature pipeline executes on wearable data with interpretable outputs and documented signal quality differences between clinical and consumer measurements.
- **Tier 2 — Signal Detection:** Model output scores show statistically meaningful deviation from personal baseline during the ninety-day window surrounding the June 2025 clinical ECG event.
- **Tier 3 — Clinical Concordance:** The ECG report confirms an irregularity consistent with the model's detection scope and the wearable data demonstrates the Tier 2 deviation pattern.

---

## Results

### Layer 1 — Cross-Validation (5-fold Stratified)

| Model | Sensitivity | Specificity | AUROC |
|---|---|---|---|
| Logistic Regression | 80.5% ± 0.1% | 84.2% ± 2.0% | 0.8817 ± 0.0066 |
| Random Forest | 80.3% ± 0.2% | 86.7% ± 1.3% | 0.8955 ± 0.0080 |
| XGBoost | 80.4% ± 0.2% | 82.5% ± 1.9% | 0.8809 ± 0.0054 |
| SVM | 80.3% ± 0.2% | 86.9% ± 1.4% | 0.8960 ± 0.0062 |

### Layer 1 — Validation Set

| Model | Sensitivity | Specificity | AUROC | AF Sensitivity | Overfitting |
|---|---|---|---|---|---|
| Logistic Regression | 80.3% | 84.5% | 0.8810 | 95.3% | None |
| Random Forest | 80.0% | 86.7% | 0.9025 | 98.8% | Significant (+13.4%) |
| XGBoost | 80.0% | 84.7% | 0.8874 | 95.3% | Significant (+13.6%) |
| **SVM** | **80.0%** | **88.1%** | **0.8981** | **98.8%** | **None (−1.2%)** |

### Selected Model — SVM

**Selection rationale:** Highest validation specificity (88.1%), fewest false positives (60), no overfitting.

**Held-out test set results (threshold = 0.34, pre-registered criterion: Sens ≥ 80%, Spec ≥ 75%):**

| Metric | Value |
|---|---|
| Sensitivity | 84.4% |
| Specificity | 87.3% |
| AUROC | 0.9080 |
| F1 Score | 0.8243 |
| AF Sensitivity | 98.8% |
| Confusion matrix | TP=265, FP=64, FN=49, TN=441 |
| Pre-registered criterion | **PASS** |

### Layer 1 — Feature Importance (Random Forest, Supplementary)

| Rank | Feature | Importance |
|---|---|---|
| 1 | RMSSD | 19.58% |
| 2 | HR Std Dev | 15.58% |
| 3 | Mean RR | 13.32% |
| 4 | HR Mean | 12.39% |
| 5 | SDNN | 11.87% |
| 6 | RR Skewness | 9.61% |
| 7 | pNN50 | 9.31% |
| 8 | RR Kurtosis | 8.32% |

### Layer 2 — Apple Watch N=1 Case Study (Complete — Null Result)

**Feature windows generated:** 1,997 (window: 30 min, step: 15 min, minimum 10 readings)
Windows by period: baseline=1,758, pre_anchor=95, post_anchor=29, follow_up=115

**Signal modality gap — KS test (all p-values ≈ 0):**

| Feature | KS Statistic | Direction | Magnitude |
|---|---|---|---|
| RMSSD | 0.424 | Apple Watch lower | Large |
| HR Mean | 0.416 | Apple Watch higher | Large |
| Mean RR | 0.390 | Apple Watch lower | Large |
| HR Std Dev | 0.359 | Apple Watch higher | Large |
| SDNN | 0.356 | Apple Watch lower | Large |
| RR Skewness | 0.297 | Apple Watch higher | Moderate |
| pNN50 | 0.210 | Apple Watch lower | Moderate |
| RR Kurtosis | 0.193 | Apple Watch lower | Moderate |

Summary: 5 large (KS > 0.3), 3 moderate (0.1–0.3), 0 small. No feature transferred cleanly between modalities.

**Probability scores:** mean=0.236, median=0.119, range=0.018–1.000

**Mann-Whitney U (one-sided, alternative=greater, pre-registered):**

| Period | n | Median Score | p-value | Effect Size r | Verdict |
|---|---|---|---|---|---|
| pre_anchor | 95 | 0.104 | 0.957 | 0.104 | Lower than baseline |
| post_anchor | 29 | 0.101 | 0.839 | 0.107 | Lower than baseline |
| follow_up | 115 | 0.121 | 0.737 | 0.035 | Higher than baseline |

**Tier assessment:** Tier 1 PASS — Tier 2 FAIL — Tier 3 FAIL

**Null result interpretation:** Two non-mutually-exclusive explanations: (1) signal modality gap — 5 of 8 features show large KS distances, placing Apple Watch windows outside the model's training distribution; (2) behavioural confound — reduced training intensity in the pre-anchor period may have shifted HRV toward more regular patterns that the model interprets as normal. The N=1 design cannot disambiguate these explanations.

---

## Project Structure

```
ga-capstone-heartbeat-or-noise/
│   README.md
│   requirements.txt
│   CLAUDE.md
│
├───data/
│   ├───physionet/            # 8,528 ECG recordings (.mat + .hea) + REFERENCE-v3.csv
│   ├───apple_watch/          # Raw CSV exports from Apple Health XML
│   │   ├───export.xml        # Source XML (1.58 GB, not committed)
│   │   ├───heart_rate_raw.csv
│   │   ├───hrv_raw.csv
│   │   ├───resting_hr_raw.csv
│   │   ├───walking_hr_raw.csv
│   │   ├───respiratory_rate_raw.csv
│   │   ├───sleep_raw.csv
│   │   ├───workouts_raw.csv
│   │   └───[additional metrics]
│   ├───simband/              # UMass Simband PPG data (pending access)
│   └───processed/
│       ├───heart_rate_clean.csv           # 478,077 records
│       ├───hrv_clean.csv                  # 5,456 records
│       ├───resting_hr_clean.csv
│       ├───walking_hr_clean.csv
│       ├───respiratory_rate_clean.csv
│       ├───physionet_features.csv         # 8,187 rows, 8 features
│       └───apple_watch_features.csv       # 1,997 windows, 12 columns
│
├───notebooks/
│   ├───01_data_exploration.ipynb          # Data acquisition, XML parsing, validation
│   ├───02_feature_engineering.ipynb       # Cleaning, artifact removal, feature extraction
│   ├───03_modelling.ipynb                 # Model training, CV, selection, evaluation
│   ├───04_layer2_analysis.ipynb           # Apple Watch case study (complete)
│   └───05_simband_validation.ipynb        # Simband validation (pending)
│
├───src/
│   ├───preprocess.py              # Apple Watch data cleaning pipeline
│   ├───features.py                # Physionet ECG feature extraction (8 HRV features)
│   ├───evaluate.py                # Model evaluation functions
│   ├───apple_watch_features.py    # Apple Watch windowed feature extraction
│   └───simband_features.py        # Simband PPG feature extraction (pending)
│
└───outputs/
    ├───figures/
    │   ├───sample_ecg.png
    │   └───aw_vs_physionet_distributions.png   # 8-subplot KS comparison
    ├───models/
    │   ├───selected_model.joblib               # SVM (threshold 0.34)
    │   ├───scaler.joblib                       # StandardScaler (fitted on training only)
    │   ├───evaluation_report.json              # Full test set metrics
    │   └───rf_feature_importance.csv           # Random Forest supplementary output
    └───layer2/
        ├───gap_quantification.csv              # 8-row KS results
        ├───probability_scores.csv              # 1,997 Apple Watch windows scored
        └───mann_whitney_results.csv            # 3-row period comparison
```

---

## Notebook Guide

| Notebook | Purpose | Status | Key Outputs |
|----------|---------|--------|-------------|
| `01_data_exploration` | Validate Physionet dataset, parse Apple Health XML, extract raw CSVs | Complete | 9 raw CSV files, `sample_ecg.png` |
| `02_feature_engineering` | Clean Apple Watch data, remove artifacts, extract Physionet features | Complete | 5 cleaned CSVs, `physionet_features.csv` |
| `03_modelling` | Train 4 classifiers, cross-validate, select model, evaluate on held-out test | Complete | `selected_model.joblib`, `scaler.joblib`, `evaluation_report.json` |
| `04_layer2_analysis` | Apple Watch windowed features, KS gap analysis, probability scoring, Mann-Whitney tests | Complete | `apple_watch_features.csv`, `gap_quantification.csv`, `probability_scores.csv`, `mann_whitney_results.csv` |
| `05_simband_validation` | Simband PPG peak detection, feature extraction, model application | Pending | — |

### Source Modules

| Module | Called By | Purpose |
|--------|-----------|---------|
| `src/preprocess.py` | `02_feature_engineering` | Cleans Apple Watch metrics: date parsing, non-numeric removal, threshold filtering, anchor period assignment |
| `src/features.py` | `02_feature_engineering` | Extracts 8 HRV features from Physionet ECG: XQRS R-peak detection, RR interval computation, quality filtering |
| `src/evaluate.py` | `03_modelling` | Confusion matrix, sensitivity, specificity, AUROC, F1, AF-specific sensitivity, threshold sweep |
| `src/apple_watch_features.py` | `04_layer2_analysis` | Windowed feature extraction from Apple Watch HR and HRV records, anchor period labelling |
| `src/simband_features.py` | `05_simband_validation` | PPG peak detection (NeuroKit2/HeartPy), RR interval extraction, 8-feature computation (pending) |

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Lucas-sam93/ga-capstone-heartbeat-or-noise.git

# Create and activate conda environment
conda create -n cvd_project python=3.13
conda activate cvd_project

# Install dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel -y
pip install wfdb xgboost joblib
```

---

## Key Methodological Decisions

| Decision | Choice | Justification |
|---|---|---|
| Primary dataset | Physionet 2017 | Clinically validated, single-lead ECG matches wearable modality |
| Noisy class | Excluded (279 records) | Signal quality failure, not rhythm classification |
| Classification type | Binary — Normal vs Abnormal | Screening requires triage decision, not diagnosis |
| Abnormal class composition | AF + Other combined | Screening tool flags any anomaly worth clinical evaluation |
| Feature constraint | 8 wearable-extractable HRV features | Ensures valid generalisation test |
| Frequency domain features | Excluded permanently | Consumer wearables do not expose raw beat intervals at consistent sampling |
| Primary metric | Sensitivity | Missing a true positive is more dangerous than a false positive in screening |
| Class imbalance | `class_weight=balanced` | Preserves training data integrity |
| Classification threshold | 0.34 (validation-optimised) | Prioritises sensitivity for screening context |
| Model selection | SVM | Highest specificity, fewest false positives, no overfitting |
| Apple Watch role | N=1 case study appendix | Null result documented; Simband is primary Layer 2 validation |
| Statistical test | Mann-Whitney U (one-sided) | Directional test matching directional Tier 2 criterion |
| Window size (Apple Watch) | 30 minutes | Exceeds 5-minute HRV minimum |
| Step size (Apple Watch) | 15 minutes | 50% overlap for smooth temporal coverage |
| RMSSD approximation (Apple Watch) | SDNN × 0.85 | Raw beat intervals unavailable from Apple Watch |
| Apple Watch merge strategy | Drop windows with no HRV record | No imputation |
| Layer 2 primary validation | UMass Simband | Multi-subject PPG with simultaneous Holter ECG ground truth |

---

## Current Status

| Layer | Status | Key Finding |
|---|---|---|
| Layer 1 — Clinical benchmark | **Complete** | SVM: Sensitivity 84.4%, Specificity 87.3%, AUROC 0.9080 |
| Layer 2 — Apple Watch (N=1) | **Complete (null result)** | No significant elevation around anchor period; large signal modality gap across 5 of 8 features |
| Layer 2 — Simband (primary) | **Pending Synapse access** | Dataset identified at syn25005551 |

**Immediate next steps:**
1. Register Synapse account and request access to syn25005551
2. Assess Simband dataset structure and implement `src/simband_features.py`
3. Apply scaler and SVM to Simband features — no retraining, no threshold adjustment
4. Retrieve June 2025 ECG report
5. Write final narrative and conclusions
6. Publish GitHub repository

---

## Limitations

This project does not constitute a clinical trial and does not produce a validated medical device. The personal Apple Watch analysis is a single-subject case study and its findings cannot be generalised to the broader population. Consumer wearable data is derived from optical heart rate sensors, which measure cardiac activity through a fundamentally different mechanism than the clinical ECG recordings used for model training. The Apple Watch SE used in this study does not have an electrical heart sensor and cannot produce ECG recordings. RMSSD values in the Apple Watch feature matrix are approximated from SDNN and should be interpreted accordingly. All personal data findings are presented as exploratory and hypothesis-generating.

---

## Acknowledgements

Dataset provided by Physionet and the Computing in Cardiology Challenge 2017 organisers. Personal health data collected via Apple Watch SE (1st Generation) and exported through Apple Health. UMass Simband dataset (pending) via the Synapse data sharing platform.
