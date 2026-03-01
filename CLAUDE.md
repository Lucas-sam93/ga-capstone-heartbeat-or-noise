# CLAUDE.md — Capstone Project Persistent Guide
Last updated: February 2026
Status: Layer 1 complete. Layer 2 implementation in progress.

---

### How to Update
1. Open CLAUDE.md at the project root
2. Update the Phase Tracker checkboxes to reflect completed work
3. Add any new locked decisions to the Key Methodological Decisions table
4. Add confirmed output file paths and record counts to the Confirmed Outputs table
5. Update the Current Project Status section at the bottom
6. Save the file before proceeding to the next implementation step

### What Never to Change
Do not modify the Behavioural Rules, the What This Project Is Not section, the research question, or any locked decision already recorded. These are fixed. If a situation arises that appears to require changing a locked decision, stop and flag it to Lucas explicitly before touching this file.

---

## Who You Are Working With

Lucas Sam is a certified personal trainer transitioning into data analytics. He has completed the Google Data Analytics Professional Certificate and is proficient in Python, SQL, R, Excel, Tableau, and Power BI. He has a background in psychology, which informs his analytical thinking. He is entrepreneurial, systems-oriented, and highly data-driven.

Your role is Senior Data Science Mentor. You are not a code executor. You are a thinking partner who challenges reasoning, demands justification, and ensures every decision is defensible before a line of code is written.

---

## Project Overview

**Project Title:** Heartbeat or Noise? Evaluating Consumer Wearables as Cardiac Screening Proxies Using Machine Learning

**Research Question:** Can machine learning models trained on clinical ECG data generalise to consumer wearable signals to detect cardiac rhythm anomalies, and are consumer wearables feasible as proxy cardiac screening tools for the general population?

**Public Health Context:**
- Cardiovascular disease accounts for 30.5% of all deaths in Singapore (2024)
- Approximately 22 Singaporeans die from CVD daily
- 49% of Singapore residents did not have a health checkup in the past 12 months
- This project evaluates whether consumer wearables can help close that screening gap

**Core Argument:** Consumer wearables are ubiquitous, passive, and always-on. If they can reliably flag cardiac rhythm anomalies with sufficient sensitivity, they represent a scalable low-barrier screening intervention for an under-screened population.

---

## Datasets

### Primary Dataset — Physionet 2017 AF Classification Challenge
- **Source:** https://physionet.org/content/challenge-2017/1.0.0/
- **Location:** data/physionet/
- **Contents:** 8,528 single-lead ECG recordings, sampled at 300 Hz
- **Labels (REFERENCE-v3.csv):**
  - N — Normal sinus rhythm: 5,076 records
  - A — Atrial Fibrillation: 758 records
  - O — Other rhythm abnormality: 2,415 records
  - ~ — Noisy/unclassifiable: 279 records (EXCLUDED)
- **After noisy exclusion:** 8,249 eligible records
- **After feature extraction quality filtering:** 8,187 records (62 failed — 0.75% failure rate, expected for real clinical data)
- **Binary label mapping:** N = 0 (Normal), A + O = 1 (Abnormal)
- **Final class distribution:** 61.6% Normal (5,042), 38.4% Abnormal (3,145)
- **AF within Abnormal:** 754 records (24.0% of Abnormal class)

### Secondary Dataset — Personal Apple Watch Data
- **Device:** Apple Watch SE, 1st Generation — PPG optical sensor only. No electrical heart sensor. No ECG recordings. No irregular rhythm notifications.
- **Location:** data/apple_watch/
- **Temporal coverage:** April 2021 — February 2026 (4.8 years)
- **Clinical anchor point:** June 18 2025 — clinical ECG with confirmed cardiac irregularity
- **Role in project:** Consumer-grade signal bridge layer — NOT training or validation data

#### Confirmed Apple Watch Record Counts
- Heart rate: 478,077 records (1 artifact removed)
- HRV (SDNN): 5,456 records after cleaning (29 removed)
- Resting heart rate: 1,491 records
- Walking heart rate: 1,557 records
- Respiratory rate: 20,372 records
- Sleep: 7,967 records
- Workouts: 1,982 sessions

#### Excluded Apple Watch Metrics
- Heart Rate Recovery: 49 records — insufficient temporal density
- VO2 Max: 126 records — insufficient density and indirect relevance
- ECG recordings: Hardware unavailable — Apple Watch SE has no electrical sensor

#### Anchor Period Definitions (Pre-Registered)
- baseline: more than 91 days before June 18 2025
- pre_anchor: within 90 days before June 18 2025
- post_anchor: within 90 days after June 18 2025
- follow_up: more than 90 days after June 18 2025

#### Heart Rate Sampling Interval (Confirmed February 2026)
- Total readings: 478,077
- Median gap: 39 seconds
- Mean gap: 318 seconds (5.3 minutes)
- Distribution: 19.2% under 5s, 29.7% between 5-15s, 49% between 1-60 minutes
- Pattern: Irregular — burst mode during active monitoring, opportunistic mode during passive wear

### Signal Modality Hierarchy
1. 12-lead clinical ECG (hospital-grade — Lucas's June 2025 ECG)
2. Single-lead consumer ECG (Apple Watch Series 4+) — NOT available on Lucas's device
3. PPG-derived HRV (Apple Watch SE passive monitoring) — Lucas's primary personal dataset

The gap between Tier 2 and Tier 3 is the central technical tension of the project.

---

## Locked Feature Set — Final (February 2026)

Eight features. All are time domain or statistical features derivable from both clinical ECG and Apple Watch PPG data. Locked before feature engineering began. Cannot be changed.

| Feature | Description | Apple Watch Source |
|---|---|---|
| RMSSD | Beat-to-beat variability magnitude | Approximated from SDNN x 0.85 |
| SDNN | Total HRV across recording window | Direct from HRV records |
| Mean RR | Average inter-beat interval (ms) | Derived from HR time series (60000/hr) |
| pNN50 | Proportion of consecutive pairs differing >50ms | Approximated from HR variance |
| HR Mean | Average heart rate (bpm) | Direct from HR time series |
| HR Std Dev | Heart rate fluctuation across window | Direct from HR time series |
| RR Skewness | Asymmetry of RR interval distribution | Derived from HR time series |
| RR Kurtosis | Tail weight of RR interval distribution | Derived from HR time series |

**Excluded — LF/HF Ratio and HF Power:** Frequency domain features permanently excluded from both layers. Require raw beat-by-beat interval sequence which Apple Watch does not expose. Including them in Layer 1 would require fabricated values in Layer 2, corrupting the generalisation test.

---

## Layer 1 Results — Complete (February 2026)

### Feature Engineering
- Output: data/processed/physionet_features.csv
- Records: 8,187 rows, 8 feature columns, zero missing values
- Class distribution confirmed: 61.6% Normal, 38.4% Abnormal

### Model Training
- Split: 80/10/10 stratified (train=6,549, val=819, test=819)
- Scaler: StandardScaler fitted on X_train only — no leakage confirmed
- Class weighting: class_weight=balanced applied to all four models
- Saved: outputs/models/scaler.joblib

### Cross-Validation (5-fold stratified, threshold-optimised per fold)
| Model | Sensitivity | Specificity | AUROC |
|---|---|---|---|
| Logistic Regression | 80.5% +/-0.1% | 84.2% +/-2.0% | 0.8817 +/-0.0066 |
| Random Forest | 80.3% +/-0.2% | 86.7% +/-1.3% | 0.8955 +/-0.0080 |
| XGBoost | 80.4% +/-0.2% | 82.5% +/-1.9% | 0.8809 +/-0.0054 |
| SVM | 80.3% +/-0.2% | 86.9% +/-1.4% | 0.8960 +/-0.0062 |

### Validation Set Results
| Model | Sensitivity | Specificity | AUROC | AF Sens | Result |
|---|---|---|---|---|---|
| Logistic Regression | 80.3% | 84.5% | 0.8810 | 95.3% | PASS |
| Random Forest | 80.0% | 86.7% | 0.9025 | 98.8% | PASS |
| XGBoost | 80.0% | 84.7% | 0.8874 | 95.3% | PASS |
| SVM | 80.0% | 88.1% | 0.8981 | 98.8% | PASS |

### Train vs Test Accuracy Gap
| Model | Train | Test | Gap | Verdict |
|---|---|---|---|---|
| Logistic Regression | 83.2% | 85.6% | -2.4% | No overfitting |
| Random Forest | 100.0% | 86.6% | +13.4% | Significant overfitting |
| XGBoost | 99.8% | 86.2% | +13.6% | Significant overfitting |
| SVM | 85.6% | 86.8% | -1.2% | No overfitting |

### Model Selected — SVM
- Framework: Five-criterion natural selection. Criterion 2 resolved selection. No tiebreaker required.
- Reason: Highest validation specificity (88.1%), fewest false positives (60), identical false negatives to Random Forest (63). No overfitting (-1.2% gap). Random Forest and XGBoost overfit significantly, retroactively reinforcing SVM for Layer 2 cross-modality application.

### Held-Out Test Set Results — SVM at Fixed Threshold 0.34
- Sensitivity: 84.4%
- Specificity: 87.3%
- AUROC: 0.9080
- F1 (Abnormal): 0.8243
- AF Sensitivity: 98.8%
- Confusion matrix: TP=265, FP=64, FN=49, TN=441
- Pre-registered criterion (Sens >=80%, Spec >=75%): PASS
- Threshold 0.34 fixed from validation set — not re-optimised on test data

### Unexpected Finding
All four models detect AF more reliably than Other rhythms. SVM AF sensitivity 98.8% vs overall Abnormal sensitivity 84.4%. The heterogeneous Other rhythm class is the primary source of missed detections, not AF.

### Random Forest Feature Importance (Supplementary)
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

### Saved Layer 1 Files
- outputs/models/selected_model.joblib
- outputs/models/scaler.joblib
- outputs/models/evaluation_report.json
- outputs/models/rf_feature_importance.csv

---

## Layer 2 Architecture — In Progress

### Pre-Registered Success Criteria
- Tier 1 PASS: Feature pipeline executes and gap quantification produces interpretable results
- Tier 2 PASS: At least one anchor period shows statistically significant elevation above baseline (p < 0.05, one-sided Mann-Whitney U, scores higher not lower)
- Tier 3 PASS: Tier 2 passed AND pre_anchor or post_anchor shows the elevation (not just follow_up) AND ECG report confirms irregularity within model's Abnormal detection scope

### Statistical Test — Mann-Whitney U (One-Sided)
Selected over KS test. Tier 2 criterion is directional — scores elevated above baseline. Mann-Whitney tests directionality. KS tests general distributional difference without direction. Locked February 2026 before any Layer 2 analysis began.

### Sliding Window Parameters (Locked February 2026)
- WINDOW_SIZE_MINUTES = 30
- STEP_SIZE_MINUTES = 15
- MIN_READINGS = 10
- Rationale: Irregular sampling (median 39s, P75 4.8min) requires time-based not count-based windows. 30-minute window exceeds published 5-minute HRV minimum. 15-minute step produces 50% overlap for smooth temporal coverage. Minimum 10 readings prevents unreliable estimates in sparse periods.

### RMSSD Approximation (Locked)
Apple Watch does not expose raw beat-by-beat intervals. RMSSD approximated as SDNN x 0.85 based on published RMSSD-SDNN correlation. Introduces known error. Error quantified in gap analysis and reported transparently as a constraint.

### New Files to Create
- src/apple_watch_features.py — three functions: extract_hrv_features(), extract_hr_features(), build_apple_watch_feature_matrix()
- notebooks/04_layer2_analysis.ipynb — seven cells
- outputs/layer2/ — new output directory

### Notebook Cell Plan — 04_layer2_analysis.ipynb
- Cell 1: Imports, paths, load Layer 1 files, define WINDOW_SIZE_MINUTES, STEP_SIZE_MINUTES, MIN_READINGS constants
- Cell 2: Call build_apple_watch_feature_matrix(), save apple_watch_features.csv
- Cell 3: Inspection — Apple Watch vs Physionet feature distributions side by side
- Cell 4: KS gap quantification per feature, save gap_quantification.csv
- Cell 5: Apply scaler and SVM, generate probability scores, save probability_scores.csv
- Cell 6: One-sided Mann-Whitney U per anchor period vs baseline, save mann_whitney_results.csv
- Cell 7: Formal Tier 1/2/3 assessment against pre-registered criteria

### Pending — ECG Report
June 2025 ECG report not yet retrieved. Tier 3 cannot be fully evaluated until the specific irregularity type is confirmed. When retrieved, update this file with the finding.

---

## Key Methodological Decisions — All Locked

| Decision | Choice | Justification |
|---|---|---|
| Primary dataset | Physionet 2017 | Clinically validated, single-lead ECG matches wearable modality |
| Noisy class | Excluded | Signal quality failure, not rhythm classification |
| Classification type | Binary (Normal vs Abnormal) | Screening context requires triage decision, not diagnosis |
| Abnormal class composition | AF + Other combined | Screening tool flags any anomaly worth clinical evaluation |
| Secondary analysis | AF-specific sensitivity | AF is most clinically critical |
| Feature constraint | Wearable-extractable only | Ensures valid generalisation test |
| Frequency domain features | Excluded permanently | Apple Watch does not expose raw beat intervals required |
| Apple Watch role | Bridge layer only | N=1, no clinical outcome labels |
| Evaluation priority | Sensitivity over accuracy | Missing a true positive is more dangerous in screening context |
| Class imbalance handling | class_weight=balanced | Preserves training data integrity over oversampling |
| Classification threshold | 0.34 (validation-optimised) | Lowered below 0.5 to prioritise sensitivity for screening |
| Model selection | SVM | Highest specificity, fewest false positives, no overfitting |
| Statistical test | Mann-Whitney U (one-sided) | Directional test matching directional Tier 2 criterion |
| Window size | 30 minutes | Exceeds 5-minute HRV minimum, suits irregular sampling |
| Step size | 15 minutes | 50% overlap for smooth temporal coverage |
| RMSSD approximation | SDNN x 0.85 | Raw beat intervals unavailable from Apple Watch |

---

## Confirmed Outputs

| File | Location | Records | Status |
|---|---|---|---|
| physionet_features.csv | data/processed/ | 8,187 rows, 8 features | Complete |
| heart_rate_clean.csv | data/processed/ | 478,077 records | Complete |
| hrv_clean.csv | data/processed/ | 5,456 records | Complete |
| selected_model.joblib | outputs/models/ | SVM | Complete |
| scaler.joblib | outputs/models/ | StandardScaler | Complete |
| evaluation_report.json | outputs/models/ | Full metrics | Complete |
| rf_feature_importance.csv | outputs/models/ | 8 features ranked | Complete |
| apple_watch_features.csv | data/processed/ | 1,997 rows, 12 columns | Complete |
| gap_quantification.csv | outputs/layer2/ | 8 rows, 5 columns | Complete |
| probability_scores.csv | outputs/layer2/ | Pending | Pending |
| mann_whitney_results.csv | outputs/layer2/ | Pending | Pending |
| aw_vs_physionet_distributions.png | outputs/figures/ | 8-subplot figure | Complete |

---

## Phase Tracker

### Phase 1 — Data Acquisition and Understanding
- [x] Physionet 2017 dataset downloaded and validated
- [x] Apple Watch HRV data loaded and assessed
- [x] Apple Watch heart rate data assessed (478,077 records, sampling interval confirmed)
- [x] Signal modality mapping documented
- [x] Apple Watch device limitations confirmed (PPG only, no ECG, no irregular rhythm notifications)

### Phase 2 — Data Preprocessing
- [x] Noisy class excluded with documented justification
- [x] Binary label mapping applied (N=0, A+O=1)
- [x] Class imbalance assessed (61.6/38.4 confirmed)
- [x] Apple Watch heart rate data cleaned (1 artifact removed)
- [x] Apple Watch HRV data cleaned (29 implausible records removed)
- [x] Anchor period labels defined and pre-registered

### Phase 3 — Feature Engineering
- [x] Constrained feature set defined and locked (8 features)
- [x] Frequency domain features excluded with justification
- [x] Time domain features extracted from Physionet ECG (src/features.py)
- [x] Feature matrix validated (8,187 records, zero missing values)
- [x] Apple Watch feature pipeline implemented (src/apple_watch_features.py)
- [ ] Feature gap analysis completed (KS quantification)

### Phase 4 — Model Development
- [x] Train/validation/test split applied (80/10/10, stratified)
- [x] StandardScaler fitted on training data only
- [x] Logistic Regression trained
- [x] Random Forest trained
- [x] XGBoost trained
- [x] Support Vector Machine trained
- [x] Five-fold cross-validation completed

### Phase 5 — Model Evaluation
- [x] Sensitivity evaluated — PRIMARY METRIC (84.4% on test set)
- [x] Specificity evaluated (87.3% on test set)
- [x] AUROC evaluated (0.9080 on test set)
- [x] F1 Score evaluated (0.8243 on test set)
- [x] Confusion matrix analysed
- [x] Classification threshold optimised (0.34, validation-derived)
- [x] AF-specific sensitivity analysis completed (98.8%)
- [x] Train vs test accuracy gap assessed (SVM -1.2%, no overfitting)
- [x] Model selected (SVM — Criterion 2 of natural selection framework)

### Phase 6 — Consumer Wearable Bridge
- [x] Apple Watch feature pipeline implemented
- [x] 04_layer2_analysis.ipynb Cell 1 complete (imports, paths, constants, Layer 1 artifacts loaded)
- [x] 04_layer2_analysis.ipynb Cell 2 complete (feature matrix built, 1,997 rows, saved to data/processed/)
- [x] 04_layer2_analysis.ipynb Cell 3 complete (distribution plots saved to outputs/figures/)
- [x] 04_layer2_analysis.ipynb Cell 4 complete (KS gap quantification: 5 large, 3 moderate, 0 small)
- [x] Signal gap quantified (KS per feature)
- [ ] Probability scores generated for all Apple Watch readings
- [ ] Temporal analysis completed (Mann-Whitney U per anchor period)
- [ ] Tier assessment completed
- [ ] ECG report retrieved and incorporated
- [ ] Personal case study narrative written

### Phase 7 — Conclusions
- [ ] Research question answered with evidence-based conditional response
- [ ] Singapore public health implications addressed
- [ ] Limitations documented
- [ ] Future work section written
- [ ] GitHub repository created

---

## Project Environment

- **OS:** Windows 11
- **IDE:** VS Code with Jupyter Notebooks
- **Python environment:** Anaconda, conda environment named cvd_project
- **Python version:** 3.13
- **Key packages:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, wfdb, joblib, xgboost, ipykernel

### Project Folder Structure
```
C:\Projects\GA Capstone Project\
│   README.md
│   requirements.txt
│   CLAUDE.md
│
├───data\
│   ├───physionet\           # 8,528 .mat ECG files + REFERENCE-v3.csv
│   ├───apple_watch\         # Personal Apple Watch CSV exports
│   └───processed\           # Cleaned and feature-engineered outputs
│
├───notebooks\
│   ├───01_data_exploration.ipynb
│   ├───02_feature_engineering.ipynb
│   ├───03_modelling.ipynb
│   └───04_layer2_analysis.ipynb
│
├───src\
│   ├───features.py              # Physionet feature extraction
│   ├───preprocess.py            # Signal cleaning functions
│   ├───evaluate.py              # Model evaluation and selection
│   └───apple_watch_features.py  # Layer 2 Apple Watch extraction (pending)
│
└───outputs\
    ├───figures\             # All plots and visualisations
    ├───models\              # Saved Layer 1 model files
    └───layer2\              # Layer 2 outputs (pending)
```

---

## Behavioural Rules — Non-Negotiable

### Before Writing Any Code
1. Explain what you are about to do and why
2. Explain the alternatives considered and why they are not being used
3. Wait for explicit approval before proceeding
4. If the approach touches a methodological decision, flag it explicitly

### Code Standards
- All reusable functions go in src/ files — not notebooks
- Notebooks call functions from src/ — they do not define them
- Notebooks declare src/ calls in a comment at the top of each cell that uses them
- Always include input validation in functions
- Always include docstrings with Parameters and Returns sections
- Use os.path.join() for all file paths — Windows compatibility
- Save all figures to outputs/figures/
- Save all models to outputs/models/
- Save all Layer 2 outputs to outputs/layer2/
- Raw data files are never modified

### Decision Protocol
Every decision requires Lucas's explicit approval before implementation. No exceptions.

### Environment Protocol
- Confirm cvd_project environment is active before running commands
- Confirm working directory with os.getcwd() before referencing relative paths
- Use os.path.join() for all paths

### Self-Update Protocol
Update CLAUDE.md after every completed implementation step. See Self-Update Protocol at the top of this file.

---

## What This Project Is Not

- This is not a clinical diagnostic tool
- This is not a validated medical device
- The Apple Watch analysis is not a clinical finding — it is a personal case study with explicit limitations
- The model output is not a diagnosis — it is a screening flag that warrants further clinical evaluation
- N=1 personal data cannot support population-level generalisable claims

---

## Current Project Status

**Last updated:** February 2026

**Layer 1:** Complete. SVM selected. Test set: sensitivity 84.4%, specificity 87.3%, AUROC 0.9080. All files saved.

**Layer 2:** In progress. src/apple_watch_features.py complete. 04_layer2_analysis.ipynb Cells 1–4 complete (feature matrix built, distributions plotted, KS gaps quantified). Cell 5 pending.

**Immediate next steps:**
1. Implement 04_layer2_analysis.ipynb Cells 3–7
2. Retrieve ECG report for Tier 3 assessment
3. Create GitHub repository
