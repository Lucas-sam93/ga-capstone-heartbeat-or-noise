# CLAUDE.md — Capstone Project Persistent Guide
Last updated: March 2026
Status: Layer 1 complete. Layer 2 Apple Watch analysis complete (null result). Layer 2 rebuild with UMass Simband pending.


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

### Secondary Dataset — Personal Apple Watch Data (N=1 Exploratory Case Study)
- **Device:** Apple Watch SE, 1st Generation — PPG optical sensor only. No electrical heart sensor. No ECG recordings. No irregular rhythm notifications.
- **Location:** data/apple_watch/
- **Temporal coverage:** April 2021 — February 2026 (4.8 years)
- **Clinical anchor point:** June 18 2025 — clinical ECG with confirmed cardiac irregularity
- **Role in project:** Exploratory N=1 case study appendix — NOT primary Layer 2 validation

#### Confirmed Apple Watch Record Counts
- Heart rate: 478,077 records (1 artifact removed)
- HRV (SDNN): 5,456 records after cleaning (29 removed)
- Resting heart rate: 1,491 records
- Walking heart rate: 1,557 records
- Respiratory rate: 20,372 records
- Sleep: 7,967 records
- Workouts: 1,982 sessions

#### Anchor Period Definitions (Pre-Registered)
- baseline: more than 91 days before June 18 2025
- pre_anchor: within 90 days before June 18 2025
- post_anchor: within 90 days after June 18 2025
- follow_up: more than 90 days after June 18 2025

#### Heart Rate Sampling Interval (Confirmed February 2026)
- Median gap: 39 seconds, Mean gap: 318 seconds
- Distribution: 19.2% under 5s, 29.7% between 5-15s, 49% between 1-60 minutes
- Pattern: Irregular — burst mode during active monitoring, opportunistic mode during passive wear

### Layer 2 Primary Dataset — UMass Simband (UMMC-DB) [PENDING ACCESS]
- **Source:** Synapse syn25005551 (registration required)
- **Device:** Samsung Simband wrist-worn smartwatch — PPG optical sensor, 128 Hz
- **Ground truth:** Simultaneous Holter ECG, cardiologist-labelled 30-second segments
- **Labels:** NSR (0), AF (1), PAC/PVC (2), Noisy (4)
- **Subjects:** 37-41 ambulatory clinic patients, approximately 10 with AF
- **Role in project:** Primary Layer 2 validation — multi-subject wearable PPG with confirmed rhythm labels
- **Status:** Dataset identified, Synapse access not yet obtained
- **Requires:** PPG peak detection step (NeuroKit2 or HeartPy) before feature extraction

### Signal Modality Hierarchy
1. 12-lead clinical ECG (hospital-grade — Lucas's June 2025 ECG)
2. Single-lead consumer ECG (Apple Watch Series 4+) — NOT available on Lucas's device
3. PPG-derived HRV (Apple Watch SE / Samsung Simband)

---

## Locked Feature Set — Final (February 2026)

Eight features. All are time domain or statistical features derivable from both clinical ECG and wearable PPG data. Locked before feature engineering began. Cannot be changed.

| Feature | Description | Apple Watch Source | Simband Source |
|---|---|---|---|
| RMSSD | Beat-to-beat variability magnitude | Approximated from SDNN x 0.85 | Computed from PPG-derived RR intervals |
| SDNN | Total HRV across recording window | Direct from HRV records | Computed from PPG-derived RR intervals |
| Mean RR | Average inter-beat interval (ms) | Derived from HR time series | Computed from PPG-derived RR intervals |
| pNN50 | Proportion of consecutive pairs differing >50ms | Approximated from HR variance | Computed from PPG-derived RR intervals |
| HR Mean | Average heart rate (bpm) | Direct from HR time series | Derived from RR intervals |
| HR Std Dev | Heart rate fluctuation across window | Direct from HR time series | Derived from RR intervals |
| RR Skewness | Asymmetry of RR interval distribution | Derived from HR time series | Computed from PPG-derived RR intervals |
| RR Kurtosis | Tail weight of RR interval distribution | Derived from HR time series | Computed from PPG-derived RR intervals |

**Excluded — LF/HF Ratio and HF Power:** Frequency domain features permanently excluded from both layers. Require raw beat-by-beat interval sequence at consistent sampling — unavailable from consumer wearables.

---

## Layer 1 Results — Complete (February 2026)

### Feature Engineering
- Output: data/processed/physionet_features.csv
- Records: 8,187 rows, 8 feature columns, zero missing values
- Class distribution: 61.6% Normal, 38.4% Abnormal

### Model Training
- Split: 80/10/10 stratified (train=6,549, val=819, test=819)
- Scaler: StandardScaler fitted on X_train only — no leakage confirmed
- Saved: outputs/models/scaler.joblib

### Cross-Validation (5-fold stratified)
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
- Framework: Five-criterion natural selection. Criterion 2 resolved selection.
- Reason: Highest validation specificity (88.1%), fewest false positives (60), no overfitting (-1.2% gap).

### Held-Out Test Set Results — SVM at Fixed Threshold 0.34
- Sensitivity: 84.4% | Specificity: 87.3% | AUROC: 0.9080 | F1: 0.8243 | AF Sensitivity: 98.8%
- Confusion matrix: TP=265, FP=64, FN=49, TN=441
- Pre-registered criterion (Sens >=80%, Spec >=75%): PASS

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

## Layer 2 — Apple Watch N=1 Case Study — Complete (March 2026)

### Apple Watch Feature Matrix
- Windows: 1,997 | baseline=1,758, pre_anchor=95, post_anchor=29, follow_up=115
- Window parameters: 30 min, 15 min step, min 10 readings
- Merge: drop windows with no HRV record — no imputation

### Gap Quantification Results (KS Test — All p-values ~0)
| Feature | KS Statistic | Direction | Band |
|---|---|---|---|
| rmssd | 0.424 | AW lower | Large |
| hr_mean | 0.416 | AW higher | Large |
| mean_rr | 0.390 | AW lower | Large |
| hr_std | 0.359 | AW higher | Large |
| sdnn | 0.356 | AW lower | Large |
| rr_skewness | 0.297 | AW higher | Moderate |
| pnn50 | 0.210 | AW lower | Moderate |
| rr_kurtosis | 0.193 | AW lower | Moderate |

Summary: 5 large (KS > 0.3), 3 moderate (0.1-0.3), 0 small. No feature transferred cleanly between modalities.

### Probability Score Summary
- Overall: mean=0.236, median=0.119, range=0.018-1.000
- baseline median=0.120 | pre_anchor median=0.104 | post_anchor median=0.101 | follow_up median=0.121

### Mann-Whitney U Results (One-Sided, Alternative=Greater)
| Period | n | Median | p-value | Effect Size r | Direction |
|---|---|---|---|---|---|
| pre_anchor | 95 | 0.104 | 0.957 | 0.104 | Lower than baseline |
| post_anchor | 29 | 0.101 | 0.839 | 0.107 | Lower than baseline |
| follow_up | 115 | 0.121 | 0.737 | 0.035 | Higher than baseline |

### Apple Watch Tier Assessment
- Tier 1: PASS — pipeline executed, gap quantification interpretable
- Tier 2: FAIL — no anchor period significantly elevated above baseline
- Tier 3: FAIL — Tier 2 not passed

### Null Result Interpretation
Two plausible non-mutually-exclusive explanations:
1. Signal modality gap — 5 of 8 features show large KS distances. Model operating outside training distribution.
2. Behavioural confound — reduced training intensity in pre-anchor period may have shifted HRV toward more regular patterns interpreted as normal by the model.

### N=1 Structural Limitation
Personal Apple Watch data is structurally weak for this validation — single individual, one event, approximated HRV, no comparison group. Null result motivates UMass Simband as primary Layer 2 validation.

---

## Layer 2 Rebuild — UMass Simband Primary Validation [PENDING]

### Rationale for Pivot
Apple Watch N=1 limitations identified during analysis. The research question is better answered with a multi-subject wearable dataset with confirmed labels. Pivot motivated by structural weakness of N=1 design, not by desire to rescue null result. Sequence documented transparently.

### Simband Implementation Plan
1. Register Synapse account and request access to syn25005551
2. Assess dataset structure and segment lengths
3. Implement PPG peak detection (NeuroKit2 or HeartPy) to extract RR intervals from 128 Hz PPG
4. Extract 8 features from RR intervals using locked feature set
5. Apply scaler and SVM — NO retraining, NO threshold adjustment
6. Evaluate sensitivity, specificity, AUROC against Holter ECG ground truth
7. Window parameters to be decided after examining Simband record structure

---

## Key Methodological Decisions — All Locked

| Decision | Choice | Justification |
|---|---|---|
| Primary dataset | Physionet 2017 | Clinically validated, single-lead ECG matches wearable modality |
| Noisy class | Excluded | Signal quality failure, not rhythm classification |
| Classification type | Binary (Normal vs Abnormal) | Screening context requires triage decision |
| Abnormal class composition | AF + Other combined | Screening tool flags any anomaly worth clinical evaluation |
| Feature constraint | Wearable-extractable only | Ensures valid generalisation test |
| Frequency domain features | Excluded permanently | Consumer wearables do not expose raw beat intervals |
| Apple Watch role | N=1 case study appendix | Null result documented; Simband is primary validation |
| Evaluation priority | Sensitivity over accuracy | Missing true positive more dangerous in screening |
| Class imbalance handling | class_weight=balanced | Preserves training data integrity |
| Classification threshold | 0.34 (validation-optimised) | Prioritises sensitivity for screening context |
| Model selection | SVM | Highest specificity, fewest false positives, no overfitting |
| Statistical test | Mann-Whitney U (one-sided) | Directional test matching directional Tier 2 criterion |
| Window size (Apple Watch) | 30 minutes | Exceeds 5-minute HRV minimum |
| Step size (Apple Watch) | 15 minutes | 50% overlap for smooth temporal coverage |
| RMSSD approximation (Apple Watch) | SDNN x 0.85 | Raw beat intervals unavailable from Apple Watch |
| Apple Watch merge strategy | Drop windows with no HRV | No imputation |
| Layer 2 primary validation | UMass Simband | Multi-subject PPG with Holter ECG ground truth |
| Mann-Whitney test direction | alternative='greater' | Pre-registered, maintained despite null finding |

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
| aw_vs_physionet_distributions.png | outputs/figures/ | 8-subplot figure | Complete |
| gap_quantification.csv | outputs/layer2/ | 8 rows | Complete |
| probability_scores.csv | outputs/layer2/ | 1,997 rows | Complete |
| mann_whitney_results.csv | outputs/layer2/ | 3 rows | Complete |
| tier_assessment.csv | outputs/layer2/ | Pending Cell 7 | Pending |

---

## Phase Tracker

### Phase 1 — Data Acquisition
- [x] Physionet 2017 downloaded and validated
- [x] Apple Watch data loaded and assessed
- [x] Signal modality mapping documented
- [x] Apple Watch device limitations confirmed

### Phase 2 — Preprocessing
- [x] Noisy class excluded
- [x] Binary label mapping applied
- [x] Apple Watch data cleaned
- [x] Anchor period labels defined and pre-registered

### Phase 3 — Feature Engineering
- [x] Feature set locked (8 features)
- [x] Frequency domain features excluded
- [x] Physionet feature matrix built (8,187 records)
- [x] Apple Watch feature pipeline implemented (src/apple_watch_features.py)
- [x] Apple Watch feature matrix built (1,997 windows)
- [x] Feature gap analysis completed (KS — 5 large, 3 moderate, 0 small)
- [ ] Simband PPG peak detection pipeline (pending access)
- [ ] Simband feature extraction (pending access)

### Phase 4 — Model Development
- [x] Stratified split applied
- [x] StandardScaler fitted on training data only
- [x] All four models trained
- [x] Five-fold cross-validation completed

### Phase 5 — Model Evaluation
- [x] All metrics evaluated
- [x] Threshold optimised (0.34)
- [x] AF-specific sensitivity completed (98.8%)
- [x] Model selected (SVM)

### Phase 6 — Consumer Wearable Bridge
- [x] Apple Watch feature pipeline implemented
- [x] Gap quantification complete
- [x] Probability scores generated
- [x] Mann-Whitney U tests complete (null result)
- [ ] Cell 7 tier assessment (pending)
- [ ] Simband access obtained
- [ ] Simband pipeline implemented and evaluated
- [ ] ECG report retrieved
- [ ] Final narrative written

### Phase 7 — Conclusions
- [ ] Research question answered
- [ ] Singapore public health implications addressed
- [ ] Limitations documented
- [ ] GitHub repository created

---

## Project Environment

- **OS:** Windows 11 | **IDE:** VS Code + Jupyter | **Environment:** Anaconda cvd_project | **Python:** 3.13
- **Key packages:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, wfdb, joblib, xgboost, ipykernel

### Project Folder Structure
```
C:\Projects\GA Capstone Project\
│   README.md | requirements.txt | CLAUDE.md
│
├───data\
│   ├───physionet\       # 8,528 .mat ECG files + REFERENCE-v3.csv
│   ├───apple_watch\     # Personal Apple Watch CSV exports
│   ├───simband\         # UMass Simband PPG data (pending)
│   └───processed\
│
├───notebooks\
│   ├───01_data_exploration.ipynb
│   ├───02_feature_engineering.ipynb
│   ├───03_modelling.ipynb
│   ├───04_layer2_analysis.ipynb       # Apple Watch case study (Cells 1-6 complete)
│   └───05_simband_validation.ipynb    # Pending
│
├───src\
│   ├───features.py
│   ├───preprocess.py
│   ├───evaluate.py
│   ├───apple_watch_features.py        # Complete
│   └───simband_features.py            # Pending
│
└───outputs\
    ├───figures\
    ├───models\
    └───layer2\
```

---

## Behavioural Rules — Non-Negotiable

### Before Writing Any Code
1. Explain what you are about to do and why
2. Explain alternatives and why they are not being used
3. Wait for explicit approval before proceeding
4. Flag any methodological decision explicitly

### Code Standards
- All reusable functions in src/ — not notebooks
- Notebooks call functions, never define them
- Notebooks declare src/ calls in comments at top of cell
- Input validation and docstrings in all functions
- os.path.join() for all paths
- Raw data files never modified

### Decision Protocol
Every decision requires Lucas's explicit approval. No exceptions.

### Environment Protocol
Confirm cvd_project active. Confirm working directory before relative paths.

---

## What This Project Is Not

- Not a clinical diagnostic tool
- Not a validated medical device
- Apple Watch analysis is not a clinical finding — it is an N=1 case study with explicit limitations
- Model output is not a diagnosis — it is a screening flag
- N=1 personal data cannot support population-level claims

---

## Current Project Status

**Last updated:** March 2026

**Layer 1:** Complete. SVM: sensitivity 84.4%, specificity 87.3%, AUROC 0.9080.

**Layer 2 — Apple Watch (N=1):** Cells 1-6 complete. Null result. Tier assessment (Cell 7) pending. Retained as case study appendix.

**Layer 2 — Simband (Primary):** Pending Synapse access to syn25005551.

**Immediate next steps:**
1. Run Cell 7 tier assessment
2. Register Synapse and request Simband access
3. Once access granted: implement src/simband_features.py and notebooks/05_simband_validation.ipynb
4. Retrieve June 2025 ECG report
5. Create GitHub repository
