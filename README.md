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

### Layer 2 Primary — MIMIC PERform AF (Zenodo)
- **Device:** Finger PPG — optical sensor, simultaneous ECG included
- **Ground truth:** Pre-assigned clinical labels at subject level — AF or NSR
- **Labels:** AF files = Abnormal (1), Non-AF files = Normal (0)
- **Subjects:** 35 critically ill adults — 19 AF, 16 NSR
- **Recording duration:** 20 minutes per subject (150,000 samples at 125 Hz)
- **Sampling rate:** 125 Hz (confirmed from metadata)
- **Source:** https://doi.org/10.5281/zenodo.6807402
- **Status:** Complete. Feature extraction, gap quantification, model inference, cross-model comparison, threshold recalibration, stress testing, and sensitivity-targeted LOOCV all executed.
- **Pivot rationale:** UMass Simband access unavailable before deadline. MIMIC PERform AF satisfies all three validation requirements: wearable PPG signal, both Normal and Abnormal subjects, pre-existing ground truth labels.

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
- **Tier 2 — Signal Detection:** Model output scores show statistically meaningful deviation from personal baseline during the ninety-day window surrounding the June 2025 clinical ECG event (Apple Watch case study only).
- **Tier 3 — Clinical Concordance:** The ECG report confirms an irregularity consistent with the model's detection scope and the wearable data demonstrates the Tier 2 deviation pattern (Apple Watch case study only).

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

**Null result interpretation:** Three non-mutually-exclusive explanations: (1) signal modality gap — 5 of 8 features show large KS distances, placing Apple Watch windows outside the model's training distribution; (2) behavioural confound — reduced training intensity in the pre-anchor period may have shifted HRV toward more regular patterns that the model interprets as normal; (3) abnormality type mismatch — the clinical ECG report confirms an intraventricular conduction delay (waveform morphology abnormality), not a rhythm irregularity. HRV features measure beat-to-beat timing variability and are structurally incapable of detecting conduction delays, which produce regular-but-abnormally-shaped beats. The N=1 design cannot disambiguate these explanations.

### Layer 2 — MIMIC PERform AF Primary Validation (Complete)

**Feature matrix:** 35 subjects (19 AF, 16 NSR), all green quality tier (≥300 PPG peaks detected). Single 20-minute window per subject. PPG nulls in 8 files handled by linear interpolation.

**Signal modality gap — KS test (all p-values ≈ 0):**

All 8 features show large KS distances (KS ≥ 0.3) — worse than Apple Watch (5 large, 3 moderate). MIMIC subjects are critically ill ICU patients with systematically higher HRV variability.

**SVM at fixed threshold 0.34:**

| Metric | Value |
|---|---|
| Sensitivity | 100% (19/19 AF) |
| Specificity | 12.5% (2/16 NSR) |
| AUROC | 0.8586 |
| Confusion matrix | TP=19, FP=14, FN=0, TN=2 |

**Tier assessment:** Tier 1 PASS — Tier 2 FAIL (specificity) — Tier 3 PASS (AUROC ≥ 0.85)

**Cross-model comparison (all four Layer 1 models, fixed thresholds):**

| Model | Threshold | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|
| Logistic Regression | 0.41 | 100% | 18.75% | 0.7368 |
| Random Forest | 0.37 | 100% | 6.25% | 0.8503 |
| XGBoost | 0.34 | 100% | 12.5% | 0.8322 |
| SVM | 0.34 | 100% | 12.5% | 0.8586 |

Specificity failure is systemic across all four models — not a model selection artefact. SVM retains highest AUROC.

**Threshold recalibration (Youden's J on MIMIC, SVM only):**

| Metric | Fixed (0.34) | Recalibrated (0.8424) |
|---|---|---|
| Sensitivity | 100% | 73.7% |
| Specificity | 12.5% | 93.8% |
| F1 | 0.7308 | 0.8235 |

Specificity recovers from 12.5% to 93.8% with domain-adapted threshold, confirming the modality gap is a calibration problem rather than a discrimination failure.

**Stress testing:**

| Test | Metric | Result |
|---|---|---|
| Bootstrap AUROC (n=1000) | Mean [95% CI] | 0.8631 [0.7246, 0.9720] |
| Bootstrap recalibrated sensitivity | Mean [95% CI] | 80.6% [57.1%, 100.0%] |
| Bootstrap recalibrated specificity | Mean [95% CI] | 91.3% [72.7%, 100.0%] |
| LOOCV Youden's J | Sens / Spec | 68.4% / 81.2% |
| Sensitivity-targeted LOOCV | Sens / Spec | 78.9% / 81.2% |

Sensitivity-targeted LOOCV achieves 78.9% sensitivity (1.1pp below pre-registered 80% criterion) with 81.2% specificity (clearing 75% criterion). The 1.1pp gap is consistent with N=19 AF leave-one-out variance. AUROC lower bound 0.7246 confirms discriminative signal persists even in unlucky resamples.

---

## BeatCheck — Cardiac Screening App (Complete)

BeatCheck is a consumer-facing web app that applies the trained SVM model to Apple Health heart rate exports. It is built as a proof-of-concept demonstration of the research pipeline in a deployable format.

**Stack:** FastAPI backend, plain HTML/CSS/JS frontend, deployed on Render.

**Input formats accepted:** Apple Health CSV export, XML export (`export.xml`), or full ZIP archive (`export.zip`). Client-side JavaScript handles large XML files via streaming to avoid browser memory limits — the full multi-year export is never transmitted to the server.

**Processing:** Uploaded data is filtered client-side to the most recent 90 days, then windowed into 30-minute overlapping windows (15-minute step). Each window is scored by the SVM at a domain-adapted threshold of 0.8368 (LOOCV mean threshold — more defensible than the point estimate of 0.8424).

**Risk output:** Percentage of windows flagged as abnormal, mapped to three tiers:

| Tier | Windows Flagged | Interpretation |
|---|---|---|
| Low | < 10% | No pattern of concern detected |
| Intermediate | 10–40% | Elevated pattern — consider follow-up |
| High | > 40% | Consistent pattern — clinical review recommended |

**Stress test result:** Real Apple Health export (4.8 years) processed successfully — Intermediate tier, 38.2% of windows flagged across 84 days analysed.

**Threshold note:** App threshold (0.8368) differs from Layer 1 threshold (0.34) by design. The 0.34 threshold was calibrated on Physionet ECG. The 0.8368 threshold is derived from LOOCV on MIMIC PERform AF PPG data and is domain-adapted for consumer wearable signals. This is not threshold shopping — it is the pre-registered recalibration strategy, applied using an out-of-sample estimate.

**Disclaimer:** A hard acknowledgement banner is displayed before upload. App output is a screening flag, not a diagnosis.

---

## Project Structure

**Note:** `data/`, `data/processed/`, and `outputs/models/` are gitignored. Large data files, personal health data, and trained model files are not committed. Clone the repo and rerun the notebooks to regenerate processed outputs and model files.

```
ga-capstone-heartbeat-or-noise/
│   README.md
│   requirements.txt          # Points to app/requirements.txt
│   CLAUDE.md
│   render.yaml               # Render deployment config
│   .python-version           # Python version pin
│
├───app/                      # BeatCheck cardiac screening app
│   ├───__init__.py
│   ├───main.py               # FastAPI backend
│   ├───pipeline.py           # ML inference pipeline
│   ├───requirements.txt      # App dependencies
│   ├───models/
│   │   ├───scaler.joblib     # Copy of outputs/models/scaler.joblib
│   │   └───selected_model.joblib
│   └───static/
│       └───index.html        # BeatCheck frontend
│
├───data/                     # gitignored — not committed
│   ├───physionet/            # 8,528 ECG recordings (.mat + .hea) + REFERENCE-v3.csv
│   ├───apple_watch/          # Raw CSV exports from Apple Health XML (personal — never committed)
│   ├───mimic_perform_af/
│   │   ├───mimic_perform_af_csv/       # 19 AF subjects — CSV + _fix.txt per subject
│   │   └───mimic_perform_non_af_csv/   # 16 NSR subjects — CSV + _fix.txt per subject
│   └───processed/            # gitignored — regenerate by running notebooks
│       ├───heart_rate_clean.csv           # 478,077 records
│       ├───hrv_clean.csv                  # 5,456 records
│       ├───physionet_features.csv         # 8,187 rows, 8 features
│       └───apple_watch_features.csv       # 1,997 windows, 12 columns
│
├───notebooks/
│   ├───01_data_exploration.ipynb          # Data acquisition, XML parsing, validation
│   ├───02_feature_engineering.ipynb       # Cleaning, artifact removal, feature extraction
│   ├───03_modelling.ipynb                 # Model training, CV, selection, evaluation
│   ├───04_layer2_analysis.ipynb           # Apple Watch case study (complete)
│   └───05_mimic_perform_af_validation.ipynb  # MIMIC PERform AF validation (complete)
│
├───src/
│   ├───preprocess.py                  # Apple Watch data cleaning pipeline
│   ├───features.py                    # Physionet ECG feature extraction (8 HRV features)
│   ├───evaluate.py                    # Model evaluation functions
│   ├───apple_watch_features.py        # Apple Watch windowed feature extraction
│   └───mimic_perform_af_features.py   # MIMIC PERform AF PPG feature extraction
│
└───outputs/
    ├───figures/
    │   ├───figures_log.csv                     # Running manifest of all figures produced
    │   ├───sample_ecg.png
    │   └───aw_vs_physionet_distributions.png   # 8-subplot KS comparison
    ├───models/                                 # gitignored — not committed
    │   ├───selected_model.joblib               # SVM (threshold 0.34)
    │   ├───scaler.joblib                       # StandardScaler (fitted on training only)
    │   ├───evaluation_report.json
    │   └───rf_feature_importance.csv
    └───layer2/
        ├───gap_quantification.csv                  # 8-row KS results (Apple Watch)
        ├───probability_scores.csv                  # 1,997 Apple Watch windows scored
        ├───mann_whitney_results.csv                # 3-row period comparison
        ├───tier_assessment.csv                     # Apple Watch tier results
        ├───mimic_perform_af_features.csv           # 35 subjects, 8 features
        ├───gap_quantification_mimic.csv            # 8-row KS results (MIMIC)
        ├───probability_scores_mimic.csv            # 35 subjects scored
        ├───tier_assessment_mimic.csv               # MIMIC tier results
        ├───cross_model_comparison_mimic.csv        # 4 models compared
        ├───threshold_recalibration_mimic.csv       # Fixed vs recalibrated threshold
        ├───stress_test_results_mimic.csv           # Bootstrap + LOOCV results
        └───sensitivity_targeted_loocv_mimic.csv    # 35-fold per-subject results
```

---

## Notebook Guide

| Notebook | Purpose | Status | Key Outputs |
|----------|---------|--------|-------------|
| `01_data_exploration` | Validate Physionet dataset, parse Apple Health XML, extract raw CSVs | Complete | 9 raw CSV files, `sample_ecg.png` |
| `02_feature_engineering` | Clean Apple Watch data, remove artifacts, extract Physionet features | Complete | 5 cleaned CSVs, `physionet_features.csv` |
| `03_modelling` | Train 4 classifiers, cross-validate, select model, evaluate on held-out test | Complete | `selected_model.joblib`, `scaler.joblib`, `evaluation_report.json` |
| `04_layer2_analysis` | Apple Watch windowed features, KS gap analysis, probability scoring, Mann-Whitney tests, tier assessment | Complete | `apple_watch_features.csv`, `gap_quantification.csv`, `probability_scores.csv`, `mann_whitney_results.csv`, `tier_assessment.csv` |
| `05_mimic_perform_af_validation` | MIMIC PERform AF PPG feature extraction, gap quantification, model application, cross-model comparison, threshold recalibration, stress testing, sensitivity-targeted LOOCV | Complete | `mimic_perform_af_features.csv`, `cross_model_comparison_mimic.csv`, `threshold_recalibration_mimic.csv`, `stress_test_results_mimic.csv` |

### Source Modules

| Module | Called By | Purpose |
|--------|-----------|---------|
| `src/preprocess.py` | `02_feature_engineering` | Cleans Apple Watch metrics: date parsing, non-numeric removal, threshold filtering, anchor period assignment |
| `src/features.py` | `02_feature_engineering` | Extracts 8 HRV features from Physionet ECG: XQRS R-peak detection, RR interval computation, quality filtering |
| `src/evaluate.py` | `03_modelling` | Confusion matrix, sensitivity, specificity, AUROC, F1, AF-specific sensitivity, threshold sweep |
| `src/apple_watch_features.py` | `04_layer2_analysis` | Windowed feature extraction from Apple Watch HR and HRV records, anchor period labelling |
| `src/mimic_perform_af_features.py` | `05_mimic_perform_af_validation` | PPG peak detection (NeuroKit2), RR interval extraction, 8-feature computation per subject |
| `app/pipeline.py` | `app/main.py` | Loads scaler and SVM, processes uploaded heart rate data, returns windowed scores and risk tier |

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Lucas-sam93/ga-capstone-heartbeat-or-noise.git

# Create and activate conda environment
conda create -n cvd_project python=3.13
conda activate cvd_project

# Install research dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel -y
pip install wfdb xgboost joblib neurokit2

# Install app dependencies (FastAPI + serving)
pip install -r app/requirements.txt
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
| Apple Watch role | N=1 case study appendix | Null result documented; MIMIC PERform AF is primary Layer 2 validation |
| Statistical test | Mann-Whitney U (one-sided) | Directional test matching directional Tier 2 criterion |
| Window size (Apple Watch) | 30 minutes | Exceeds 5-minute HRV minimum |
| Step size (Apple Watch) | 15 minutes | 50% overlap for smooth temporal coverage |
| RMSSD approximation (Apple Watch) | SDNN × 0.85 | Raw beat intervals unavailable from Apple Watch |
| Apple Watch merge strategy | Drop windows with no HRV record | No imputation |
| Layer 2 primary validation | MIMIC PERform AF (Zenodo) | Real PPG, pre-labeled binary ground truth, no access barrier, deadline-compatible |
| Window size (MIMIC PERform AF) | Single 20-minute window per subject | Full recording duration — preserves Layer 1 pipeline contract, longer window strengthens HRV reliability |
| Label mapping (MIMIC PERform AF) | AF files = 1, Non-AF files = 0 | Pre-assigned at file level, maps directly to binary framing |
| PPG null handling (MIMIC PERform AF) | Linear interpolation pre-peak-detection | 8 of 35 files affected, worst case 1.15% — standard PPG cleaning practice |
| App input format | Apple Health CSV/XML/ZIP export | Most common consumer wearable export format, proven pipeline from Apple Watch analysis |
| App data truncation | Client-side JS filters to most recent 90 days | Solves file size problem at source — full multi-year export never transmitted to server |
| App window strategy | Percentage of windows above threshold | Mirrors clinical Holter monitoring, neutralises single-window false positive problem |
| App risk tiers | 3-band: Low <10%, Intermediate 10–40%, High >40% | Clinically grounded, consistent with ACC/AHA screening frameworks |
| App backend | FastAPI | Native Python, minimal boilerplate, serves joblib directly |
| App threshold | 0.8368 (LOOCV mean) | Out-of-sample estimate — more defensible than point estimate 0.8424 |
| App disclaimer style | Hard acknowledgement banner | User must acknowledge before upload — responsible design for general public |

---

## Current Status

| Layer | Status | Key Finding |
|---|---|---|
| Layer 1 — Clinical benchmark | **Complete** | SVM: Sensitivity 84.4%, Specificity 87.3%, AUROC 0.9080 |
| Layer 2 — Apple Watch (N=1) | **Complete (null result)** | No significant elevation around anchor period; large signal modality gap across 5 of 8 features |
| Layer 2 — MIMIC PERform AF (primary) | **Complete** | AUROC 0.8586; sensitivity-targeted LOOCV: 78.9% sensitivity, 81.2% specificity after threshold recalibration |
| BeatCheck App | **Complete** | FastAPI + HTML/CSS/JS, deployed on Render. Accepts CSV/XML/ZIP. Stress-tested — Intermediate tier, 38.2% windows flagged |

**Key conclusion:** Modality gap is the central obstacle to direct generalisation from ECG to PPG. AUROC confirms discriminative signal transfers across modalities. Threshold recalibration recovers specificity from 12.5% to 81.2%+ in LOOCV, demonstrating the failure is calibration-based rather than a discrimination failure.

**ECG Report (June 2025):** Retrieved. Abnormal ECG confirmed — sinus rhythm with intraventricular conduction delay and ST abnormality (probable anterior early repolarisation). The conduction delay is a waveform morphology abnormality, not a rhythm irregularity, providing a mechanistic explanation for the Apple Watch null result.

---

## Conclusion

### Answering the Research Question

*Can machine learning models trained on clinical ECG data generalise to consumer wearable signals to detect cardiac rhythm anomalies, and are consumer wearables feasible as proxy cardiac screening tools for the general population?*

**The answer is a qualified yes.** The discriminative signal trained on clinical ECG data transfers meaningfully to consumer wearable PPG signals, but direct threshold generalisation fails due to the modality gap between clinical and consumer measurement systems.

Layer 1 established that an SVM classifier trained on 8 HRV features from 8,187 clinical ECG recordings achieves 84.4% sensitivity and 87.3% specificity (AUROC 0.9080) on held-out test data — surpassing the pre-registered criterion of sensitivity ≥80% at specificity ≥75%. The model demonstrates strong discriminative ability for distinguishing normal from abnormal cardiac rhythms using only features extractable from consumer wearables.

Layer 2 tested whether this performance transfers to real wearable data. The MIMIC PERform AF validation (N=35) revealed the central finding: AUROC 0.8586 confirms the model retains genuine discriminative ability across modalities, but the fixed ECG-calibrated threshold (0.34) produces 100% sensitivity at only 12.5% specificity. This is not a discrimination failure — it is a calibration failure. When the threshold is recalibrated for the PPG domain (0.8368 via LOOCV), sensitivity-targeted LOOCV recovers 78.9% sensitivity at 81.2% specificity, approaching the pre-registered criterion within the variance expected from N=35 leave-one-out evaluation.

The Apple Watch N=1 case study produced a null result — probability scores were not elevated around the clinical anchor event. Three non-mutually-exclusive explanations account for this: the signal modality gap, a behavioural confound from reduced training intensity, and critically, the nature of the clinical finding itself. The ECG report from the National Heart Centre Singapore confirms an intraventricular conduction delay — a waveform morphology abnormality that affects QRS shape, not beat-to-beat timing. HRV features, which measure inter-beat interval variability, are structurally incapable of detecting this type of abnormality. This finding narrows the project's validated scope to rhythm irregularities specifically, while demonstrating that real-world cardiac abnormalities encompass a broader spectrum than HRV-based screening can capture.

### Implications for Singapore Public Health

With 49% of Singapore residents foregoing annual health checkups and cardiovascular disease claiming approximately 22 lives daily, the screening gap is real and consequential. Consumer wearables are already worn by millions and continuously collect cardiac data without requiring active participation.

This project demonstrates that the bridge from clinical models to consumer wearable signals is feasible but not yet direct. The discriminative signal transfers (AUROC 0.86 across modalities), but deployment requires domain-adapted thresholds calibrated to the specific wearable measurement system. A clinical ECG threshold applied directly to PPG data will over-flag — producing unnecessary anxiety and clinical referrals. A recalibrated threshold restores clinically meaningful specificity.

For Singapore's public health strategy, the practical path forward involves: (1) training or adapting models with PPG-specific calibration data from the target population; (2) partnering with wearable manufacturers to access richer signal data (raw RR intervals rather than summary statistics); and (3) validating on larger, population-representative cohorts rather than ICU patients. The technology is not ready for autonomous screening today, but the discriminative foundation is present — the engineering problem is calibration, not capability.

---

## Limitations

This project does not constitute a clinical trial and does not produce a validated medical device. The personal Apple Watch analysis is a single-subject case study and its findings cannot be generalised to the broader population. Consumer wearable data is derived from optical heart rate sensors, which measure cardiac activity through a fundamentally different mechanism than the clinical ECG recordings used for model training. The Apple Watch SE used in this study does not have an electrical heart sensor and cannot produce ECG recordings. RMSSD values in the Apple Watch feature matrix are approximated from SDNN and should be interpreted accordingly. The MIMIC PERform AF validation uses 35 critically ill ICU subjects — a population that differs from the general screening target. Threshold recalibration and LOOCV results were computed on the same 35 subjects and should be interpreted as indicative rather than definitive. The locked 8-feature HRV set detects rhythm irregularities (beat-to-beat timing abnormalities) but cannot detect morphological abnormalities such as conduction delays or ST changes, which affect waveform shape rather than inter-beat timing — as demonstrated by the clinical ECG report findings. All personal data findings are presented as exploratory and hypothesis-generating.

---

## Acknowledgements

Dataset provided by Physionet and the Computing in Cardiology Challenge 2017 organisers. Personal health data collected via Apple Watch SE (1st Generation) and exported through Apple Health. MIMIC PERform AF dataset provided via Zenodo (doi.org/10.5281/zenodo.6807402).
