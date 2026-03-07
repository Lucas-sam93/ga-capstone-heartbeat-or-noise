# CLAUDE.md — Capstone Project Persistent Guide
Last updated: March 2026
Status: Layer 1 complete. Layer 2 Apple Watch analysis complete (null result). Layer 2 MIMIC PERform AF validation complete including cross-model comparison, threshold recalibration, stress testing, and sensitivity-targeted LOOCV. Research loop closed. Narrative pending.


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

### Layer 2 Primary Dataset — MIMIC PERform AF (Zenodo)
- **Source:** https://doi.org/10.5281/zenodo.6807402
- **Device:** Finger PPG — optical sensor, simultaneous ECG included
- **Ground truth:** Pre-assigned clinical labels at subject level — AF or NSR
- **Labels:** AF files = Abnormal (1), Non-AF files = Normal (0)
- **Subjects:** 35 critically ill adults — 19 AF, 16 NSR
- **Recording duration:** 20 minutes per subject (confirmed — 150,000 samples ÷ 125 Hz = 1,200s)
- **Rows per file:** 150,001 (150,000 data points)
- **Columns:** Time, PPG, ECG, resp (all float64)
- **Sampling rate:** 125 Hz (confirmed from _fix.txt metadata)
- **PPG column:** Present in all 35 files — pipeline-ready
- **Nulls:** 8 of 35 files have PPG nulls — worst case NAF_012 at 1,728 nulls (1.15%). Handled by linear interpolation pre-peak-detection.
- **resp column:** Absent in 9 of 35 files — irrelevant, pipeline uses PPG only
- **Companion files:** _fix.txt per subject — contains subject ID, signal descriptions, sampling rate
- **File formats available:** CSV, MATLAB .mat, WFDB
- **Format selected:** CSV — most straightforward for Python pipeline
- **Role in project:** Primary Layer 2 validation — real PPG with pre-labeled binary ground truth, no access barrier
- **Status:** Complete. Feature extraction, gap quantification, model inference, and tier assessment all executed.
- **Pivot rationale:** Simband access unavailable before deadline. MIMIC PERform AF satisfies all three validation requirements: wearable PPG signal, both Normal and Abnormal subjects, pre-existing ground truth labels.
- **Requires:** PPG peak detection step (NeuroKit2, sampling_rate=125) to extract RR intervals before feature extraction.

### UMass Simband — SUPERSEDED
- **Status:** Access not obtained before deadline. Replaced by MIMIC PERform AF.
- **Retained for reference only.** Do not attempt to reactivate as primary dataset.

### Signal Modality Hierarchy
1. 12-lead clinical ECG (hospital-grade — Lucas's June 2025 ECG)
2. Single-lead consumer ECG (Apple Watch Series 4+) — NOT available on Lucas's device
3. PPG-derived HRV (Apple Watch SE / MIMIC PERform AF finger PPG)

---

## Locked Feature Set — Final (February 2026)

Eight features. All are time domain or statistical features derivable from both clinical ECG and wearable PPG data. Locked before feature engineering began. Cannot be changed.

| Feature | Description | Apple Watch Source | MIMIC PERform AF Source |
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
Personal Apple Watch data is structurally weak for this validation — single individual, one event, approximated HRV, no comparison group. Null result motivated pivot to MIMIC PERform AF as primary Layer 2 validation.

---

## Layer 2 Rebuild — MIMIC PERform AF Primary Validation

### Rationale for Pivot
Apple Watch N=1 limitations identified during analysis. The research question is better answered with a multi-subject wearable dataset with confirmed labels. Simband access unavailable before deadline. MIMIC PERform AF (Zenodo) selected as deadline-compatible alternative satisfying all three validation requirements. Pivot motivated by structural weakness of N=1 design and access constraints — not by desire to rescue null result. Sequence documented transparently.

### MIMIC PERform AF Implementation Plan
1. Load all 35 CSV files — 19 AF, 16 NSR
2. Apply linear interpolation to PPG nulls in 8 affected files before any processing
3. Implement PPG peak detection (NeuroKit2, sampling_rate=125) to extract RR intervals
4. Extract 8 features from RR intervals using locked feature set — single 20-minute window per subject
5. Apply scaler with transform() only — NO fit_transform(), NO retraining
6. Apply SVM at fixed threshold 0.34 — NO threshold adjustment
7. Evaluate sensitivity, specificity, AUROC against pre-assigned AF/NSR labels
8. Run gap quantification (KS test) against Physionet training distribution

---

## Layer 2 — MIMIC PERform AF Primary Validation — Complete (March 2026)

### Feature Matrix
- Subjects: 35 (19 AF = Abnormal, 16 NSR = Normal)
- Quality tier: All 35 subjects green (≥300 peaks detected). Zero amber, zero red, zero skipped.
- Window: Single 20-minute window per subject
- Output: outputs/layer2/mimic_perform_af_features.csv — 35 rows, 11 columns (subject_id, label, quality_tier + 8 features)

### Gap Quantification (KS Test vs Physionet Training Distribution)
All 8 features show LARGE KS distances (KS ≥ 0.3). All p-values ~0.

| Feature | KS Statistic | Direction |
|---|---|---|
| rr_skewness | 0.5186 | MIMIC higher |
| pnn50 | 0.4739 | MIMIC higher |
| rr_kurtosis | 0.4121 | MIMIC higher |
| mean_rr | 0.3853 | MIMIC lower |
| hr_mean | 0.3853 | MIMIC higher |
| hr_std | 0.3373 | MIMIC higher |
| sdnn | 0.3236 | MIMIC higher |
| rmssd | 0.3747 | MIMIC higher |

Summary: 8 large (KS > 0.3), 0 moderate, 0 small. Worse than Apple Watch (5 large, 3 moderate). MIMIC subjects (critically ill ICU patients) show systematically higher HRV variability — clinically plausible.

### Probability Score Summary
- Mean: 0.8027 | Median: 0.8414
- Predicted Abnormal: 33 of 35 | Predicted Normal: 2 of 35
- Model pushing almost all subjects toward Abnormal — modality gap producing systematic score inflation

### Evaluation Results — SVM at Fixed Threshold 0.34
- Sensitivity: 100% (19/19 AF cases correctly flagged — zero false negatives)
- Specificity: 12.5% (2/16 NSR correct, 14/16 false positives)
- AUROC: 0.8586
- F1: 0.7308
- Confusion matrix: TP=19, FP=14, FN=0, TN=2

### Tier Assessment
- Tier 1 (pipeline execution): PASS
- Tier 2 (Sens ≥80% AND Spec ≥75%): FAIL — specificity 12.5%
- Tier 3 (AUROC ≥0.85): PASS — AUROC 0.8586

### Interpretation
100% sensitivity with 12.5% specificity is coherent and mechanistically explained. The KS gap quantification directly predicts this outcome — all 8 features shifted toward values the model associates with Abnormal. Threshold 0.34 was calibrated on Physionet ECG. When applied to a population whose baseline features already appear abnormal to the model, almost everything crosses threshold.

AUROC 0.86 is the critical finding: the model retains genuine discriminative ability across modalities. The specificity failure is a threshold calibration problem caused by the modality gap — not a failure of the underlying model to distinguish rhythm patterns.

### Cross-Model Comparison (Cell 7)
All four Layer 1 models run on MIMIC PERform AF at fixed Layer 1 thresholds.

| Model | Threshold | Sensitivity | Specificity | AUROC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.41 | 100% | 18.75% | 0.7368 | 0.7451 |
| Random Forest | 0.37 | 100% | 6.25% | 0.8503 | 0.7170 |
| XGBoost | 0.34 | 100% | 12.5% | 0.8322 | 0.7308 |
| SVM | 0.34 | 100% | 12.5% | 0.8586 | 0.7308 |

Key finding: Specificity failure is systemic across all four models — not a model selection problem. SVM retains highest AUROC (0.8586). Model selection confirmed correct.

### Threshold Recalibration (Cell 8)
Youden's J optimal threshold on MIMIC PERform AF — SVM only.

| Metric | Fixed (0.34) | Recalibrated (0.8424) |
|---|---|---|
| Sensitivity | 100% | 73.7% |
| Specificity | 12.5% | 93.8% |
| F1 | 0.7308 | 0.8235 |
| TP/FP/FN/TN | 19/14/0/2 | 14/1/5/15 |

Key finding: Specificity recovers from 12.5% to 93.8% with domain-adapted threshold. Optimal threshold 0.8424 vs fixed 0.34 — gap quantifies magnitude of modality shift. Caveat: threshold found and evaluated on same 35 subjects — stress tested in Cell 9.

### Stress Testing (Cell 9)
Three stress tests on SVM MIMIC inference.

Test 1 — Bootstrap AUROC (n=1000):
- Mean AUROC: 0.8631
- 95% CI: [0.7246, 0.9720]

Test 2 — Bootstrap Recalibrated Metrics (n=1000):
- Sensitivity: 80.6% [57.1%, 100.0%]
- Specificity: 91.3% [72.7%, 100.0%]
- F1: 0.8534 [0.7059, 0.9631]

Test 3 — LOOCV Youden's J:
- Sensitivity: 68.4% (TP=13, FN=6)
- Specificity: 81.2% (TN=13, FP=3)
- F1: 0.7429

Key finding: AUROC lower bound 0.7246 confirms discriminative signal is present even in unlucky resamples. LOOCV specificity 81.2% clears pre-registered 75% criterion. LOOCV sensitivity 68.4% does not clear 80% criterion — N=35 limitation acknowledged.

### Sensitivity-Targeted LOOCV (Cell 10)
LOOCV repeated with sensitivity-first threshold criterion (tpr >= 0.80 on training fold).

Per-fold threshold stats:
- Mean: 0.8368 | Min: 0.8367 | Max: 0.8414 | Fallback folds: 0 of 35

| Method | Sensitivity | Specificity | F1 |
|---|---|---|---|
| Youden's J LOOCV | 68.4% | 81.2% | 0.7429 |
| Sensitivity-targeted LOOCV | 78.9% | 81.2% | 0.8108 |
| Pre-registered criterion | ≥80% | ≥75% | — |

Key finding: Sensitivity improves 10.5 percentage points (68.4% to 78.9%) with no specificity cost. Both methods hold specificity at 81.2%, clearing pre-registered criterion. Sensitivity 1.1pp below criterion — consistent with N=19 AF variance in leave-one-out evaluation. Sensitivity-targeted LOOCV is the primary reported figure for screening context. Youden's J reported alongside for transparency.

### Known Warning
sklearn UserWarning during inference: scaler and model fitted with feature names but received array without feature names. Cosmetic only — does not affect output values. No fix required.

### Complete Layer 2 Picture
Two datasets, consistent story:
- Apple Watch (N=1): Null result. Scores not elevated above baseline. Explained by modality gap and N=1 structural limitations.
- MIMIC PERform AF (N=35): 100% sensitivity, 12.5% specificity, AUROC 0.86. Model catches every abnormal case but threshold miscalibration from modality gap produces excessive false positives.

Both point to the same conclusion: modality gap is the central obstacle to direct generalisation. AUROC confirms discriminative signal is present. Domain adaptation or threshold recalibration would likely recover specificity.

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
| Apple Watch role | N=1 case study appendix | Null result documented; MIMIC PERform AF is primary validation |
| Evaluation priority | Sensitivity over accuracy | Missing true positive more dangerous in screening |
| Class imbalance handling | class_weight=balanced | Preserves training data integrity |
| Classification threshold | 0.34 (validation-optimised) | Prioritises sensitivity for screening context |
| Model selection | SVM | Highest specificity, fewest false positives, no overfitting |
| Statistical test | Mann-Whitney U (one-sided) | Directional test matching directional Tier 2 criterion |
| Window size (Apple Watch) | 30 minutes | Exceeds 5-minute HRV minimum |
| Step size (Apple Watch) | 15 minutes | 50% overlap for smooth temporal coverage |
| RMSSD approximation (Apple Watch) | SDNN x 0.85 | Raw beat intervals unavailable from Apple Watch |
| Apple Watch merge strategy | Drop windows with no HRV | No imputation |
| Layer 2 primary validation | MIMIC PERform AF (Zenodo) | Real PPG, pre-labeled binary ground truth, no access barrier, deadline-compatible |
| Mann-Whitney test direction | alternative='greater' | Pre-registered, maintained despite null finding |
| Window size (MIMIC PERform AF) | Single 20-minute window per subject | Confirmed recording duration is 20 min — preserves Layer 1 pipeline contract, longer window strengthens HRV feature reliability |
| Label mapping (MIMIC PERform AF) | AF files = 1 (Abnormal), Non-AF files = 0 (Normal) | Pre-assigned at file level, no derivation needed, maps directly to binary framing |
| PPG null handling (MIMIC PERform AF) | Linear interpolation pre-peak-detection | 8 of 35 files affected, worst case 1.15% — standard PPG cleaning practice, documentable and defensible |
| App input format | Apple Health CSV export (heart rate time series) | Most common consumer wearable export format, proven pipeline exists from Apple Watch analysis |
| App window strategy | Percentage of windows above threshold | Most honest representation, mirrors clinical Holter monitoring, neutralises single-window false positive problem |
| App risk tiers | 3-band: Low <10% flagged, Intermediate 10-40%, High >40% | Clinically grounded, reduces false positive panic, consistent with ACC/AHA screening frameworks |
| App data truncation | Client-side JavaScript filters to most recent 90 days before upload | Solves file size problem at source — full multi-year export never transmitted to server |
| App backend | FastAPI | Native Python, minimal boilerplate, serves joblib directly |
| App frontend | Plain HTML/CSS/JS served by FastAPI | Consumer-facing product appearance for demo presentation |

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
| tier_assessment.csv | outputs/layer2/ | Apple Watch — 3 rows | Complete |
| figures_log.csv | outputs/figures/ | Running manifest of all figures | Created on first figure output |
| gap_quantification_mimic.csv | outputs/layer2/ | 8 rows | Complete |
| probability_scores_mimic.csv | outputs/layer2/ | 35 rows | Complete |
| tier_assessment_mimic.csv | outputs/layer2/ | 3 rows | Complete |
| cross_model_comparison_mimic.csv | outputs/layer2/ | 4 rows | Complete |
| threshold_recalibration_mimic.csv | outputs/layer2/ | 1 row | Complete |
| stress_test_results_mimic.csv | outputs/layer2/ | Stress test results | Complete |
| sensitivity_targeted_loocv_mimic.csv | outputs/layer2/ | 35 rows | Complete |
| sensitivity_targeted_comparison_mimic.csv | outputs/layer2/ | 2 rows | Complete |

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
- [x] MIMIC PERform AF CSV structure verified (125 Hz, PPG column confirmed, 150,000 samples per file)
- [x] MIMIC PERform AF PPG peak detection pipeline implemented (src/mimic_perform_af_features.py)
- [x] MIMIC PERform AF feature extraction complete (35 subjects, all green tier)

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
- [x] Cell 7 tier assessment complete (Apple Watch — Tier 1 PASS, Tier 2 FAIL, Tier 3 FAIL)
- [x] MIMIC PERform AF dataset downloaded from Zenodo
- [x] MIMIC PERform AF CSV structure verified
- [x] MIMIC PERform AF pipeline implemented and evaluated
- [x] src/mimic_perform_af_features.py complete
- [x] notebooks/05_mimic_perform_af_validation.ipynb complete (Cells 1-6)
- [x] Cross-model comparison on MIMIC (all four Layer 1 models, fixed thresholds)
- [x] Threshold recalibration ROC analysis (Youden's J)
- [x] Stress testing — bootstrap AUROC, bootstrap recalibrated metrics, LOOCV
- [x] Sensitivity-targeted LOOCV
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
│   ├───physionet\           # 8,528 .mat ECG files + REFERENCE-v3.csv
│   ├───apple_watch\         # Personal Apple Watch CSV exports
│   ├───mimic_perform_af\    # MIMIC PERform AF CSV files (Zenodo — downloaded)
│   └───processed\
│
├───notebooks\
│   ├───01_data_exploration.ipynb
│   ├───02_feature_engineering.ipynb
│   ├───03_modelling.ipynb
│   ├───04_layer2_analysis.ipynb                # Apple Watch case study — Complete
│   └───05_mimic_perform_af_validation.ipynb    # MIMIC PERform AF validation — Complete
│
├───src\
│   ├───features.py
│   ├───preprocess.py
│   ├───evaluate.py
│   ├───apple_watch_features.py                 # Complete
│   └───mimic_perform_af_features.py            # Complete
│
├───app\                                        # Cardiac screening web app — In progress
│   ├───main.py
│   ├───pipeline.py
│   ├───requirements.txt
│   ├───models\                                 # Copy scaler.joblib and selected_model.joblib here
│   └───static\
│       └───index.html
│
└───outputs\
    ├───figures\
    │   └───figures_log.csv                     # Running manifest — created on first figure output
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

### Claude Code Reporting Format 
Report back in this format only:
STATUS: pass/fail
COUNTS: [key numbers only]
WARNINGS: [verbatim only, none if clean]

### Code Standards
- All reusable functions in src/ — not notebooks
- Notebooks call functions, never define them
- Notebooks declare src/ calls in comments at top of cell
- Input validation and docstrings in all functions
- os.path.join() for all paths
- Raw data files never modified

### Figure Logging Standards — Non-Negotiable
Every figure produced must satisfy all of the following without exception:

**Saving**
- Saved to outputs/figures/ with a descriptive filename: {analysis_step}_{description}_{year}.png
- Example: mimic_roc_curve_svm_lr_comparison_2026.png
- Example: mimic_feature_distributions_threeway_2026.png

**Manifest entry**
- Every saved figure is logged as a new row in outputs/figures/figures_log.csv
- Columns: filename | description | notebook | cell_reference | analysis_step | date_produced
- If figures_log.csv does not exist, create it before saving the first figure

**Notebook documentation**
- Every cell that produces a figure must have a markdown cell directly above it containing:
  - Section heading
  - What the figure shows
  - Why it was produced (which analysis step it supports)
  - Expected interpretation

**Styling consistency**
- All figures use the same colour palette, font sizes, and axis label conventions
- Figures saved at 150 dpi minimum
- All axes labelled with units where applicable
- All figures include a title

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

**Layer 2 — Apple Watch (N=1):** Complete. Null result — pre_anchor scores not elevated above baseline. Tier 1 PASS, Tier 2 FAIL, Tier 3 FAIL. Retained as exploratory case study appendix.

**Layer 2 — MIMIC PERform AF (Primary):** Complete. 35 subjects, all green tier. Sensitivity 100%, Specificity 12.5%, AUROC 0.8586. Tier 1 PASS, Tier 2 FAIL (specificity), Tier 3 PASS. Modality gap (8 large KS distances) explains threshold miscalibration. AUROC confirms discriminative signal transfers across modalities. Research loop closed.

**App — Cardiac Screening Web App:** Architecture locked. Pending narrative completion.

**Immediate next steps:**
1. Retrieve June 2025 ECG report
2. Write final narrative
3. Execute app build via Claude Code
4. Create GitHub repository
